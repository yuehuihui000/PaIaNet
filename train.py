import os, torch, random, pickle, time
from argparse import ArgumentParser
import numpy as np
import datetime
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel.scatter_gather import gather
from parallel import DataParallelModel, DataParallelCriterion
import cv2
import dataset
from utils import SalEval, AverageMeterSmooth, Logger, plot_training_process

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = ArgumentParser()
parser.add_argument('--data_dir', default='./dataset/', type=str, help='data directory')
parser.add_argument('--width', default=320, type=int, help='width of RGB image')
parser.add_argument('--height', default=320, type=int, help='height of RGB image')
parser.add_argument('--max_epochs', default=100, type=int, help='max number of epochs')
parser.add_argument('--num_workers', default=4, type=int, help='No. of parallel threads')
parser.add_argument('--batch_size', default=6, type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--warmup', default=0, type=int, help='lr warming up epoches')
parser.add_argument('--scheduler', default='poly', type=str, choices=['step', 'poly', 'cos'],
                    help='Lr scheduler (valid: step, poly, cos)')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for multi-step lr decay')
parser.add_argument('--milestones', default='[30, 60, 90]', type=str, help='milestones for multi-step lr decay')
parser.add_argument('--print_freq', default=50, type=int, help='frequency of printing training info')
parser.add_argument('--savedir', default='./Results', type=str, help='Directory to save the results')
parser.add_argument('--resume', default="", type=str, help='use this checkpoint to continue training')
parser.add_argument('--pretrained', default="", type=str, help='path for the ImageNet pretrained backbone model')
parser.add_argument('--seed', default=666, type=int, help='Random Seed')
parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to run on the GPU')
parser.add_argument('--model', default='Models.model', type=str, help='which model to test')

args = parser.parse_args()

exec('from {} import Net as net'.format(args.model))

cudnn.benchmark = False
cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

#----------->  lr <-------------#
def adjust_lr(optimizer, epoch):
    if epoch < args.warmup:
        lr = args.lr * (epoch + 1) / args.warmup
    else:
        if args.scheduler == 'cos':
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.max_epochs))
        elif args.scheduler == 'poly':
            lr = args.lr * (1 - epoch * 1.0 / args.max_epochs) ** 0.9
        elif args.scheduler == 'step':
            lr = args.lr
            for milestone in eval(args.milestones):
                if epoch >= milestone:
                    lr *= args.gamma
        else:
            raise ValueError('Unknown lr mode {}'.format(args.scheduler))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#----------->  loss <-------------#
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, target):
        loss = F.binary_cross_entropy_with_logits(inputs, target)
        return loss
    
@torch.no_grad()
def val(val_loader, epoch):
    model.eval()
    salEvalVal = SalEval()

    total_batches = len(val_loader)
    for iter, (input, target, edge, name) in enumerate(val_loader):
        if args.gpu:
            input = input.cuda()
            target = target[:,0,:,:].cuda()
            edge = edge[:,0,:,:].cuda()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)
        edge = torch.autograd.Variable(edge)
        
        
        start_time = time.time()
        output_edge, output_location, output = model(input)
        
        output_edge = F.interpolate(output_edge, size=target.size()[1:], mode='bilinear', align_corners=True)
        output_location = F.interpolate(output_location, size=target.size()[1:], mode='bilinear', align_corners=True)
        output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        
        pred_location = torch.sigmoid(output_location[0,0]).cpu().numpy()*255
        pred = torch.sigmoid(output[0,0]).cpu().numpy()*255

        #cv2.imwrite(osp.join('./Outputs/LLI-TE', name[0]), pred)

        torch.cuda.synchronize()
        val_times.update(time.time() - start_time)

        val_loss = criterion(output[:,0,:,:], target.float()) + criterion(output_location[:,0,:,:], target.float()) + criterion(output_edge[:,0,:,:], edge.float()) \
                 +F.l1_loss(1-torch.sigmoid(output[:,0,:,:]), 1-target.float())

        val_losses.update(val_loss.item())

        # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)
        salEvalVal.addBatch(torch.sigmoid(output[:, 0, :, :]), target.bool())

        if iter % args.print_freq == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Time: %.3f loss: %.3f (avg: %.3f) loss_location: %.3f loss_final: %.3f' %
                        (epoch, args.max_epochs, iter, total_batches, val_times.avg,
                        val_losses.val, val_losses.avg, criterion(output_location[:,0,:,:], target.float()), criterion(output[:,0,:,:], target.float())))

    F_beta_max, MAE, tp, tr, F_beta  = salEvalVal.getMetric()

    return F_beta_max, MAE


def train(train_loader, epoch, verbose=True):
    model.train()
    if verbose:
        salEvalTrain = SalEval()

    total_batches = len(train_loader)
    end = time.time()
    for iter, (input, target, edge, name) in enumerate(train_loader):
        if args.gpu == True:
            input = input.cuda()
            target = target[:,0,:,:].cuda()
            edge = edge[:,0,:,:].cuda()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)
        edge = torch.autograd.Variable(edge)

        start_time = time.time()
        # run the model
        output_edge, output_location, output = model(input)       
        output_edge = F.interpolate(output_edge, size=target.size()[1:], mode='bilinear', align_corners=True)
        output_location = F.interpolate(output_location, size=target.size()[1:], mode='bilinear', align_corners=True)
        output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        # loss
        loss = criterion(output[:,0,:,:], target.float()) + criterion(output_location[:,0,:,:], target.float()) + criterion(output_edge[:,0,:,:], edge.float())\
             +F.l1_loss(1-torch.sigmoid(output[:,0,:,:]), 1-target.float())

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item())
        train_batch_times.update(time.time() - start_time)
        train_data_times.update(start_time - end)
        record['avgtrain_losses'].append(train_losses.avg)

        if verbose:
            # compute the confusion matrix
            if args.gpu and torch.cuda.device_count() > 1:
                output = gather(output, 0, dim=0)
            salEvalTrain.addBatch(torch.sigmoid(output[:, 0, :, :]), target.bool())

        if iter % args.print_freq == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Batch time: %.3f Data time: %.3f ' \
                        'loss: %.3f (avg: %.3f) loss_final: %.3f  lr: %.1e' % \
                        (epoch, args.max_epochs, iter, total_batches, \
                         train_batch_times.avg, train_data_times.avg, \
                         train_losses.val, train_losses.avg, criterion(output[:,0,:,:], target.float()), lr))
        end = time.time()

    if verbose:
        F_beta_max, MAE, tp, tr, F_beta  = salEvalTrain.getMetric()
        record['train']['F_beta'].append(F_beta_max)
        record['train']['MAE'].append(MAE)

        return F_beta_max, MAE


# create the directory if not exist
if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)

log_name = 'log_' + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + '.txt'
logger = Logger(os.path.join(args.savedir, log_name))
logger.info('Called with args:')
for (key, value) in vars(args).items():
    logger.info('{0:16} | {1}'.format(key, value))

#----------->  load the model <-------------#
model = net()
if args.gpu and torch.cuda.device_count() > 1:
    model = DataParallelModel(model)
if args.gpu:
    model = model.cuda()

logger.info('Model Architecture:\n' + str(model))
total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
logger.info('Total network parameters: ' + str(total_paramters))

logger.info('Data statistics:')
#logger.info('mean: [%.5f, %.5f, %.5f], std: [%.5f, %.5f, %.5f]' % (*data['mean'], *data['std']))

criterion = CrossEntropyLoss()
if args.gpu and torch.cuda.device_count() > 1 :
    criterion = DataParallelCriterion(criterion)
if args.gpu:
    criterion = criterion.cuda()

train_losses = AverageMeterSmooth()
train_batch_times = AverageMeterSmooth()
train_data_times = AverageMeterSmooth()
val_losses = AverageMeterSmooth()
val_times = AverageMeterSmooth()

record = {
        'lr': [], 'avgtrain_losses': [],
        'val': {'F_beta1': [], 'MAE1': [], 'F_beta2': [], 'MAE2': [], 'F_beta3': [], 'MAE3': []},
        'train': {'F_beta': [], 'MAE': []}
        }
bests = {'F_beta_tr': 0., 'F_beta_val1': 0., 'F_beta_val2': 0., 'F_beta_val3': 0., 'MAE_tr': 1., 'MAE_val1': 1., 'MAE_val2': 1., 'MAE_val3': 1.}


train_set = dataset.Dataset(args.data_dir, 'train.txt', transform=None)
val_set1 = dataset.Dataset(args.data_dir, 'LLI-TE1.txt', transform=None)
val_set2 = dataset.Dataset(args.data_dir, 'LLI-TE2.txt', transform=None)
val_set3 = dataset.Dataset(args.data_dir, 'LLI-TE3.txt', transform=None)
train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
val_loader1 = torch.utils.data.DataLoader(
        val_set1, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
        )
val_loader2 = torch.utils.data.DataLoader(
        val_set2, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
        )
val_loader3 = torch.utils.data.DataLoader(
        val_set3, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
        )


optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, eval(args.milestones), args.gamma)
logger.info('Optimizer Info:\n' + str(optimizer))

start_epoch = 0
if args.resume is not None:
    if os.path.isfile(args.resume):
        logger.info('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        logger.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        logger.info('=> no checkpoint found at {}'.format(args.resume))

for epoch in range(start_epoch, args.max_epochs):
    # train for one epoch
    lr = adjust_lr(optimizer, epoch)
    record['lr'].append(lr)
    length = len(train_loader)

    F_beta_tr, MAE_tr = train(train_loader, epoch, verbose=True)

    # evaluate on validation set
    F_beta_val1, MAE_val1 = val(val_loader1, epoch)
    record['val']['F_beta1'].append(F_beta_val1)
    record['val']['MAE1'].append(MAE_val1)
    
    F_beta_val2, MAE_val2 = val(val_loader2, epoch)
    record['val']['F_beta2'].append(F_beta_val2)
    record['val']['MAE2'].append(MAE_val2)
    
    F_beta_val3, MAE_val3 = val(val_loader3, epoch)
    record['val']['F_beta3'].append(F_beta_val3)
    record['val']['MAE3'].append(MAE_val3)
    
    if F_beta_tr > bests['F_beta_tr']: bests['F_beta_tr'] = F_beta_tr
    if MAE_tr < bests['MAE_tr']: bests['MAE_tr'] = MAE_tr
    if F_beta_val1 > bests['F_beta_val1']: bests['F_beta_val1'] = F_beta_val1
    if MAE_val1 < bests['MAE_val1']: bests['MAE_val1'] = MAE_val1
    if F_beta_val2 > bests['F_beta_val2']: bests['F_beta_val2'] = F_beta_val2
    if MAE_val2 < bests['MAE_val2']: bests['MAE_val2'] = MAE_val2
    if F_beta_val3 > bests['F_beta_val3']: bests['F_beta_val3'] = F_beta_val3
    if MAE_val3 < bests['MAE_val3']: bests['MAE_val3'] = MAE_val3

    scheduler.step()
    torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_F_beta1': bests['F_beta_val1'],
            'best_MAE1': bests['MAE_val1'],
            'best_F_beta2': bests['F_beta_val2'],
            'best_MAE2': bests['MAE_val2'],
            'best_F_beta3': bests['F_beta_val3'],
            'best_MAE3': bests['MAE_val3']
            }, os.path.join(args.savedir, 'checkpoint.pth'))

    # save the model also
    model_file_name = os.path.join(args.savedir, 'model_epoch' + str(epoch) + '.pth')
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_file_name)

    logger.info('Epoch %d: F_beta (tr) %.4f (Best: %.4f) MAE (tr) %.4f (Best: %.4f) ' 'F_beta1 (val) %.4f (Best: %.4f) MAE1 (val) %.4f (Best: %.4f)  F_beta2 (val) %.4f (Best: %.4f) MAE2 (val) %.4f (Best: %.4f)  F_beta3 (val) %.4f (Best: %.4f) MAE3 (val) %.4f (Best: %.4f)' %  (epoch, F_beta_tr, bests['F_beta_tr'], MAE_tr, bests['MAE_tr'], F_beta_val1, bests['F_beta_val1'], MAE_val1, bests['MAE_val1'], F_beta_val2, bests['F_beta_val2'], MAE_val2, bests['MAE_val2'], F_beta_val3, bests['F_beta_val3'], MAE_val3, bests['MAE_val3']))
    plot_training_process(record, args.savedir, bests)

logger.close()
