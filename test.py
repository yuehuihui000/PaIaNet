import torch, cv2, time, os
from PIL import Image
import os.path as osp
import numpy as np
import pickle
import scipy
import skimage.io as io
import torch.nn.functional as F
from argparse import ArgumentParser
from Models.model import Net as net
import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = ArgumentParser()
parser.add_argument('--data_dir', default='./dataset/', type=str, help='data directory')
parser.add_argument('--file_list', default='ALL', type=str, help='dataset list',
                    choices=['LLI-TE1', 'LLI-TE2', 'LLI-TE3'])
parser.add_argument('--width', default=320, type=int, help='width of RGB image')
parser.add_argument('--height', default=320, type=int, help='height of RGB image')
parser.add_argument('--savedir', default='./Outputs', type=str, help='directory to save the results')
parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='Run on CPU or GPU. If TRUE, then GPU')
parser.add_argument('--pretrained', default='./Results/model_epoch100.pth', type=str, help='PRNet, training with LLI_resize_new')
args = parser.parse_args()

#model
model = net()
state_dict = torch.load(args.pretrained)
if list(state_dict.keys())[0][:7] == 'module.':
    state_dict = {key[7:]: value for key, value in state_dict.items()}
model.load_state_dict(state_dict['state_dict'], strict=True)

print('Model resumed from %s' % args.pretrained)

if args.gpu:
    model = model.cuda()
model.eval()

if args.file_list == 'ALL':
    args.file_list = ['LLI-TE1', 'LLI-TE2', 'LLI-TE3']
else: print("dataset error !!")
    
test_set1 = dataset.Dataset(args.data_dir, 'LLI-TE1.txt', transform=None)
test_loader1 = torch.utils.data.DataLoader(test_set1, batch_size= 1, shuffle=False, num_workers= 4, pin_memory=True)

test_set2 = dataset.Dataset(args.data_dir, 'LLI-TE2.txt', transform=None)
test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size= 1, shuffle=False, num_workers= 4, pin_memory=True)

test_set3 = dataset.Dataset(args.data_dir, 'LLI-TE3.txt', transform=None)
test_loader3 = torch.utils.data.DataLoader(test_set3, batch_size= 1, shuffle=False, num_workers= 4, pin_memory=True)

Test_loader = [test_loader1, test_loader2, test_loader3]
        
for e in range(len(args.file_list)):
    file_list = args.file_list[e]
    test_loader = Test_loader[e]

    if not osp.isdir(osp.join(args.savedir, file_list)):
        os.mkdir(osp.join(args.savedir, file_list))

    for i, (image, target, edge, name) in enumerate(test_loader):
        image = image.cuda()
        image = torch.autograd.Variable(image)

        with torch.no_grad():
            output_edge, output_location, output = model(image)
            pred0 = torch.sigmoid(output[0,0]).cpu().numpy()*65536.0
            pred = pred0.astype('uint16')
            pred = np.where(pred>256*50, pred, 0)
            
        io.imsave(osp.join(args.savedir, file_list, name[0]), pred)
print("Done!")
