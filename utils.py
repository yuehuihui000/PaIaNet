import os
import numpy as np
import torch
import scipy.io as sio
import logging
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SalEval(object):
    def __init__(self, nthresh=100):
        self.nthresh = nthresh
        self.thresh = torch.from_numpy(np.linspace(0, 1.0 - 1e-10, nthresh)).float().cuda()
        self.EPSILON = 1e-8

        self.recall = torch.zeros(nthresh).cuda()
        self.precision = torch.zeros(nthresh).cuda()
        self.mae = 0
        self.num = 0

    @torch.no_grad()
    def addBatch(self, predict, gth):
        for t in range(self.nthresh):
            bi_res = predict > self.thresh[t]
            #print('bi_res',bi_res.type())
            intersection = torch.sum(torch.sum(bi_res & gth, dim=1), dim=1).float()
            all_gth = torch.sum(torch.sum(gth, dim=1), dim=1).float()
            all_pred = torch.sum(torch.sum(bi_res, dim=1), dim=1).float()
            self.recall[t] += torch.sum(intersection / (all_gth + self.EPSILON))
            self.precision[t] += torch.sum(intersection / (all_pred + self.EPSILON))
        self.mae += torch.sum(torch.abs(gth.float() - predict)) / (gth.shape[1] * gth.shape[2])
        self.num += gth.shape[0]

    @torch.no_grad()
    def getMetric(self):

        tr = self.recall / self.num
        tp = self.precision / self.num
        MAE = self.mae / self.num
        F_beta = (1 + 0.3) * tp * tr / (0.3 * tp + tr + self.EPSILON)

        return torch.max(F_beta).cpu().item(), MAE.cpu().item(), tp.cpu(), tr.cpu(), F_beta.cpu()


class Logger(object):
    def __init__(self, path='log.txt'):
        self.logger = logging.getLogger('Logger')
        self.file_handler = logging.FileHandler(path, 'w')
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeterSmooth(object):
    '''Computes and stores the average and current value'''
    def __init__(self, maxlen=100):
        self.reset()
        self.maxlen = maxlen

    def reset(self):
        self.memory = []
        self.avg = 0
        self.val = 0
        self.count = 0

    def update(self, val):
        if self.count >= self.maxlen:
            self.memory.pop(0)
            self.count -= 1
        self.memory.append(val)
        self.val = val
        self.sum = sum(self.memory)
        self.count += 1
        self.avg = self.sum / self.count


def plot_training_process(record, save_dir, bests):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes[0, 0].plot(record['avgtrain_losses'], linewidth=1.)
    axes[0, 0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 0].legend(['avg train_losses'], loc='upper right')
    axes[0, 0].set_xlabel('Iter')
    axes[0, 0].set_ylabel('avg train_losses')
    
    axes[1, 0].plot(record['lr'], linewidth=1.)
    axes[1, 0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 0].legend(['learning rate'], loc='upper right')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('learning rate')

    axes[0, 1].plot(record['val']['F_beta1'], linewidth=1., color='blue')
    axes[0, 1].plot(record['val']['F_beta2'], linewidth=1., color='red')
    axes[0, 1].plot(record['val']['F_beta3'], linewidth=1., color='green')
    axes[0, 1].plot(record['train']['F_beta'], linewidth=1., color='orange')
    axes[0, 1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 1].legend(['F_beta_val1 (Best: %.4f)' % bests['F_beta_val1'], 'F_beta_val2 (Best: %.4f)' % bests['F_beta_val2'], 'F_beta_val3 (Best: %.4f)' % bests['F_beta_val3'],  'F_beta_tr (Best: %.4f)' % bests['F_beta_tr']], loc='lower right')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F_beta')

    axes[1, 1].plot(record['val']['MAE1'], linewidth=1., color='blue')
    axes[1, 1].plot(record['val']['MAE2'], linewidth=1., color='red')
    axes[1, 1].plot(record['val']['MAE3'], linewidth=1., color='green')
    axes[1, 1].plot(record['train']['MAE'], linewidth=1., color='orange')
    axes[1, 1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 1].legend(['MAE_val1 (Best: %.4f)' % bests['MAE_val1'], 'MAE_val2 (Best: %.4f)' % bests['MAE_val2'], 'MAE_val3 (Best: %.4f)' % bests['MAE_val3'], 'MAE_tr (Best: %.4f)' % bests['MAE_tr']], loc='upper right')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'record.pdf'))
    plt.close(fig)