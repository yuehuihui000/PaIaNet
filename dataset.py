import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
class Dataset(Dataset):
    def __init__(self, data_dir, dataset, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = list()
        self.msk_list = list()
        self.emsk_list = list()
        with open(osp.join(self.data_dir, dataset), 'r') as lines:
            for line in lines:
                line_arr = line.split()
                self.img_list.append(osp.join(self.data_dir, 'image', line_arr[0].strip()))
                self.msk_list.append(osp.join(self.data_dir, 'mask',  line_arr[0].strip()))
                self.emsk_list.append(osp.join(self.data_dir, 'edge',  line_arr[0].strip()))


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]).resize((320,320)).convert('RGB')
        label = Image.open(self.msk_list[idx]).resize((320,320)).convert('L')
        edge = Image.open(self.emsk_list[idx]).resize((320,320)).convert('L')
        name = self.msk_list[idx].split('/')[-1]

        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)
        edge  = transforms.ToTensor()(edge)
        
        return image, label, edge, name