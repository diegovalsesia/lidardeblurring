import torch
import numpy as np
import os
from PIL import Image
import random
from scipy.io import loadmat
import torch.nn.functional as F
import cv2

# this is deblur+denoise
class KernelDeblurData(torch.utils.data.Dataset):
    def __init__(self, opt, img_path, gt_path=None, is_patch=False):
        super(KernelDeblurData, self).__init__()
        self.img_li = [path for path in os.listdir(img_path)]
        self.inp_path = img_path
        self.gt_path = gt_path
        self.is_patch = is_patch
        self.opt = opt
        ker_mat = loadmat("./Levin09_v7.mat")
        ker_mat = ker_mat['kernels']
        self.get_ker = lambda idx: ker_mat[0, idx]
    def __getitem__(self, index):
        if not self.gt_path:
            rgb = Image.open(os.path.join(self.inp_path, self.img_li[index]))
            rgb = np.array(rgb)
            if self.is_patch==True:
                rgb = self.get_patch_pair(rgb,self.opt.config['train']['is_large'])
            blur = self.generate_blur(rgb)
            patch_size = self.opt.config['train']['patch_size']
            blur = blur[patch_size//2: patch_size//2 + patch_size, patch_size//2: patch_size//2 + patch_size,:]
            rgb = rgb[patch_size//2: patch_size//2 + patch_size, patch_size//2: patch_size//2 + patch_size,:]
            rgb = rgb.transpose(2, 0, 1)
            rgb = rgb.astype(np.float32) / 255
            rgb = torch.Tensor(rgb)
            rgb = rgb.to(self.opt.device)
            blur = blur.transpose(2, 0, 1)
            blur = blur.astype(np.float32) / 255
            blur = torch.Tensor(blur)
            rgb = rgb.to(self.opt.device)
            blur = blur.to(self.opt.device)
            return blur, rgb, self.img_li[index].split('.')[0]
        else:
            blur = Image.open(os.path.join(self.inp_path, self.img_li[index]))
            blur = np.array(blur).transpose([2, 0, 1])
            blur = blur.astype(np.float32) / 255
            blur = torch.Tensor(np.array(blur))
            blur = blur.to(self.opt.device)
            #####
            #blur = blur + 10/255.0*torch.randn(blur.shape, device = self.opt.device) 
            ####
            rgb = Image.open(os.path.join(self.gt_path,self.img_li[index]))
            rgb = np.array(rgb).transpose([2, 0, 1])
            rgb = rgb.astype(np.float32) / 255
            rgb = torch.Tensor(np.array(rgb))
            rgb = rgb.to(self.opt.device)
            return blur, rgb, self.img_li[index].split('.png')[0]
    def __len__(self):
        return len(self.img_li)
    
    def get_patch_pair(self, gt, is_large):
        
        hb,wb,_ = gt.shape

        gt_patch_size = int(self.opt.config['train']['patch_size'] * 2)
        gt_h_start = random.randint(0, hb - gt_patch_size)
        gt_w_start = random.randint(0, wb -  gt_patch_size)

        gt_patch = gt[gt_h_start: gt_h_start + gt_patch_size, gt_w_start: gt_w_start + gt_patch_size,:]
        # blur_patch = blur[:, gt_h_start: gt_h_start + gt_patch_size, gt_w_start: gt_w_start + gt_patch_size]

        return  gt_patch
    
    # blur+noise
    def generate_blur(self,rgb):
        ker = self.get_ker(random.randint(0,7))
        h, w, _ = rgb.shape
        scale = 1440/255
        hk, wk = ker.shape
        ker = cv2.resize(ker, (int(hk * scale), int(wk * scale)), interpolation=cv2.INTER_LINEAR)
        ker /= ker.sum()
        blur = cv2.filter2D(rgb, -1, ker)
        #blur = blur + np.random.normal(0, 10, blur.shape)
        return blur