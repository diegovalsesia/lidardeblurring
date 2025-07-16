import torch
import numpy as np
import os
from PIL import Image
import random
import torch.nn.functional as F
from scipy.io import loadmat
from torchvision.transforms import GaussianBlur
import cv2
import torch.nn as nn
import time

class KernelDepthDeblurData(torch.utils.data.Dataset):
    def __init__(self, opt, img_path, gt_path=None, depth_path=None, is_patch=False, gt_depth_path=None):
        super(KernelDepthDeblurData,self).__init__()
        self.img_li = [path for path in os.listdir(img_path)]
        self.opt = opt
        self.img_path = img_path
        self.depth_path = depth_path
        self.gt_path = gt_path
        self.is_patch = is_patch
        ker_mat = loadmat("./Levin09_v7.mat")
        ker_mat = ker_mat['kernels']
        self.get_ker = lambda idx: ker_mat[0, idx]
        self.gt_depth_path = gt_depth_path
        self.patch_size = self.opt.config['train']['patch_size']
           
    def __getitem__(self, index):
        if not self.gt_path:
            rgb = Image.open(os.path.join(self.img_path, self.img_li[index]))
            rgb = np.array(rgb)
            depth= Image.open(os.path.join(self.depth_path, self.img_li[index]))
            depth = np.array(depth)
            depth = depth.astype(np.float32) / 1000
            depth = depth/ np.max(depth)
            if self.gt_depth_path !=None:
                gt_depth = Image.open(os.path.join(self.gt_depth_path, self.img_li[index]))
                gt_depth = np.array(gt_depth)
                gt_depth = gt_depth.astype(np.float32) / 1000
                mask = np.where(gt_depth>0.01,1,0)
                gt_depth = torch.Tensor(gt_depth).to(self.opt.device).unsqueeze(0)
                mask = torch.Tensor(mask).to(self.opt.device).unsqueeze(0)
                rgb = torch.cat((rgb,gt_depth,mask),dim=0)
            if self.is_patch==True:
                depth, rgb = self.get_patch_pair(depth, rgb)
            blur = self.generate_blur(rgb)
            blur = blur[ self.patch_size//2: self.patch_size//2 + self.patch_size, self.patch_size//2: self.patch_size//2 + self.patch_size,:]
            rgb = rgb[self.patch_size//2: self.patch_size//2 + self.patch_size, self.patch_size//2: self.patch_size//2 + self.patch_size,:]
            rgb = rgb.transpose(2, 0, 1)
            rgb = rgb.astype(np.float32) / 255
            rgb = torch.Tensor(rgb)
            rgb = rgb.to(self.opt.device)
            blur = blur.transpose(2, 0, 1)
            blur = blur.astype(np.float32) / 255
            blur = torch.Tensor(blur)
            blur = blur.to(self.opt.device)
            dh,dw = depth.shape
            depth = depth[dh // 4 - 1: int( dh // 4 * 3),  dw  // 4 - 1: int(dw // 4 * 3)]
            depth = torch.Tensor(np.array(depth)).unsqueeze(0)
            depth = depth.to(self.opt.device)
            return blur, depth, rgb, self.img_li[index].split('.')[0]
        else:
            blur = Image.open(os.path.join(self.img_path, self.img_li[index]))
            blur = np.array(blur).transpose([2, 0, 1])
            blur = blur.astype(np.float32) / 255
            blur = torch.Tensor(np.array(blur))
            blur = blur.to(self.opt.device)
            ####
            #blur = blur + 10/255.0*torch.randn(blur.shape, device = self.opt.device) 
            ###
            rgb = Image.open(os.path.join(self.gt_path,self.img_li[index]))
            rgb = np.array(rgb).transpose([2, 0, 1])
            rgb = rgb.astype(np.float32) / 255
            rgb = torch.Tensor(np.array(rgb))
            rgb = rgb.to(self.opt.device)
            depth= Image.open(os.path.join(self.depth_path, self.img_li[index]))
            depth = np.array(depth)
            depth = depth.astype(np.float32) / np.max(depth)
            depth = torch.Tensor(np.array(depth)).unsqueeze(0)
            depth = depth.to(self.opt.device)
            return blur, depth, rgb, self.img_li[index].split('.png')[0]
        
    def __len__(self):
        return len(self.img_li)
    
    def get_patch_pair(self,dep, gt):
                
        hb,wb,_ = gt.shape
        scale = 8
        dep = cv2.resize(dep,(wb//scale,hb//scale))
        dep_h, dep_w = dep.shape # c h w
        dep_patch_size = int(self.patch_size// scale *2)
        dep_h_start = random.randint(0, dep_h - dep_patch_size)
        dep_w_start = random.randint(0, dep_w - dep_patch_size)
        dep_patch = dep[dep_h_start: dep_h_start + dep_patch_size, dep_w_start: dep_w_start + dep_patch_size]
        gt_patch_size = int(self.patch_size *2)
        gt_h_start = int(dep_h_start * scale)
        gt_w_start = int(dep_w_start * scale)
        gt_patch = gt[gt_h_start: gt_h_start + gt_patch_size, gt_w_start: gt_w_start + gt_patch_size,:]
       # blur_patch = blur[:, gt_h_start: gt_h_start + gt_patch_size, gt_w_start: gt_w_start + gt_patch_size]

        return  dep_patch, gt_patch
    
    def generate_blur(self,rgb):
        ker = self.get_ker(random.randint(0,7))
        h, w, _ = rgb.shape
        scale = 1440/255
        hk, wk = ker.shape
        ker = cv2.resize(ker, (int(hk * scale), int(wk * scale)), interpolation=cv2.INTER_LINEAR)
        ker /= ker.sum()
        blur = cv2.filter2D(rgb, -1, ker)
        ## 
        #blur = blur + np.random.normal(0, 10, blur.shape)
        ## 
        return blur
    
    def generate_ker(self):
        ker = self.get_ker(random.randint(0,7))
        scale = 1440/255
        hk, wk = ker.shape
        ker = cv2.resize(ker, (int(hk * scale), int(wk * scale)), interpolation=cv2.INTER_LINEAR)
        ker /= ker.sum()
        ker = torch.Tensor(ker).unsqueeze(0)
        ker = torch.cat((ker,ker,ker),dim = 0)
        return ker
