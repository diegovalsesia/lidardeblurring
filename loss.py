import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, inp, target):
        return ((nn.functional.mse_loss(inp, target, reduction='none') + self.eps2) ** .5).mean()


class OutlierAwareLoss(nn.Module):
    def __init__(self, kernel_size=False):
        super(OutlierAwareLoss, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size)
        self.kernel = kernel_size

    def forward(self, out, lab):
        b, c, h, w = out.shape
        p = self.kernel // 2
        delta = out - lab
        if self.kernel:
            delta_ = torch.nn.functional.pad(delta, (p, p, p, p))
            patch = self.unfold(delta_).reshape(b, c,
                                                self.kernel, self.kernel,
                                                h, w).detach()
            var = patch.std((2, 3)) / (2 ** .5)
            avg = patch.mean((2, 3))
        else:
            var = delta.std((2, 3), keepdims=True) / (2 ** .5)
            avg = delta.mean((2, 3), True)
        weight = 1 - (-((delta - avg).abs() / var)).exp().detach()
        # weight = 1 - (-1 / var.detach()-1/ delta.abs().detach()).exp()
        loss = (delta.abs() * weight).mean()
        return loss



class LossDepthDeblur(nn.Module):
    def __init__(self):
        super(LossDepthDeblur, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean())
        return loss


class LossDeblur(nn.Module):
    def __init__(self):
        super(LossDeblur, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean())
        return loss


class LossKernelDepthDeblur(nn.Module):
    def __init__(self,opt):
        super(LossKernelDepthDeblur, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()
        self.loss_l1 = nn.L1Loss()

    def forward(self, out, gt):
        if gt.size(1) > 3:
            gt_dep = gt[:,3:,:,:]
            gt = gt[:,:3,:,:]
            loss = (self.loss_l1(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean() + 0.001 * self.loss_dep(gt, gt_dep).mean() )
        else:
            loss = (self.loss_l1(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean() )
        return loss



def import_loss(training_task, opt):
    if training_task == 'DepthDeblur':
        return LossDepthDeblur()
    elif training_task == 'Deblur':
        return LossDeblur()
    elif training_task =='KernelDepthDeblur' or training_task =="KernelDeblur":
        return LossKernelDepthDeblur(opt)
    elif training_task == 'warmup':
        return LossWarmup()
    else:
        raise ValueError('unknown task, please choose from [DepthDeblur, Deblur].')
