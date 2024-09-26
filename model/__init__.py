import torch
import os
from importlib import import_module
from torch.nn.parallel import DistributedDataParallel as DDP 
from  torch.nn.parallel import DataParallel
from .archs.restormer import Restormer
from .archs.depthrestormer import DepthRestormer
from .archs.NAFNet_arch import NAFNet
from .archs.DepthNAFNet import DepthNAFNet
from .archs.DeblurDiNATL import NADeblurL
from .archs.depthDeblurDiNATL import DepthNADeblurL
from .archs.stripformer import Stripformer
from .archs.depthstripformer import DepthStripformer
_all__ = {
    'import_model',
    'Restormer',
    'NADeblurL',
    'Stripformer',
    'NAFNet',
    'DepthRestormer',
    'DepthNAFNet',
    'DepthNADeblurL',
    'DepthNAFNet',

    'import_module'
}


def import_model(opt,gpu_id=None):
    model_name = opt.config['model']['name']
    if  model_name not in _all__:
        raise ValueError('unknown model, please choose from [ Restormer, NADeblurL, Stripformer, NAFNet, DepthRestormer, DepthNAFNet, DepthNADeblurL]')
    model = getattr(import_module('model'),model_name)()
    
    if opt.config['model']['resume']:
        model.load_state_dict(torch.load(opt.config['model']['pretrained'],map_location=opt.device)['model_state_dict'],strict=True)
    elif opt.config['model']['pretrained']:
        model.load_state_dict(torch.load(opt.config['model']['pretrained'],map_location=opt.device)['params'],strict=False)

    model = model.to(opt.device)
    if opt.config['model']['num_gpus'] > 1:
        model =DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    return model
