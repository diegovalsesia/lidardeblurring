import torch
import numpy as np
import cv2
import os
from PIL import Image
import random
import torch.nn.functional as F

class DepthDeblurData(torch.utils.data.Dataset):
    def __init__(self, opt, blur_path, sharp_path=None, depth_path=None, is_patch=False):
        super(DepthDeblurData,self).__init__()
        self.img_li = [path for path in os.listdir(blur_path)]
        self.blur_path = blur_path
        self.depth_path = depth_path
        self.sharp_path = sharp_path
        self.opt = opt
        self.is_patch = is_patch
        self.patch_size = self.opt.config['train']['patch_size']
           

    def __getitem__(self, index):
        blur = Image.open(os.path.join(self.blur_path, self.img_li[index]))
        blur = np.array(blur).transpose([2, 0, 1])
        blur = blur.astype(np.float32) / 255
        blur = torch.Tensor(np.array(blur))
        blur = blur.to(self.opt.device)
        depth= Image.open(os.path.join(self.depth_path, self.img_li[index].split('_ker')[0]+'.png'))
        depth = np.array(depth)
        depth = depth.astype(np.float32) / np.max(depth)
        depth = torch.Tensor(np.array(depth)).unsqueeze(0)
        depth = depth.to(self.opt.device)


        if self.sharp_path: # gt_path -> train/test not demo
            rgb = Image.open(os.path.join(self.sharp_path, self.img_li[index].split('_ker')[0]+'.png'))
            rgb = np.array(rgb).transpose([2, 0, 1])
            rgb = rgb.astype(np.float32) / 255
            rgb = torch.Tensor(np.array(rgb))
            rgb = rgb.to(self.opt.device)
            if self.is_patch==True:
                depth, rgb,blur = self.get_patch_pair(blur, depth, rgb,self.opt.config['train']['is_large'])
            return blur, depth, rgb, self.img_li[index].split('.')[0]

        return blur, self.img_li[index].split('.')[0]

    def __len__(self):
        return len(self.img_li)
    
    def get_patch_pair(self, blur,dep, gt, is_large):
                    
        _,hb,wb = gt.shape
        scale = 8
        # dep = cv2.resize(dep,(wb//8,hb//8))
        dep = F.interpolate(dep.unsqueeze(0),size=(hb//8,wb//8)).squeeze(0)
        _,dep_h, dep_w = dep.shape # c h w
        dep_patch_size = int(self.patch_size// scale)
        dep_h_start = random.randint(0, dep_h - dep_patch_size)
        dep_w_start = random.randint(0, dep_w - dep_patch_size)
        dep_patch = dep[:,dep_h_start: dep_h_start + dep_patch_size, dep_w_start: dep_w_start + dep_patch_size]
        gt_patch_size = self.patch_size
        gt_h_start = int(dep_h_start * scale)
        gt_w_start = int(dep_w_start * scale)
        gt_patch = gt[:,gt_h_start: gt_h_start + gt_patch_size, gt_w_start: gt_w_start + gt_patch_size]
        blur_patch = blur[:, gt_h_start: gt_h_start + gt_patch_size, gt_w_start: gt_w_start + gt_patch_size]
        return  dep_patch, gt_patch, blur_patch