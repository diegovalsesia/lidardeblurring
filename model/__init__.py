import torch
import os
from importlib import import_module
from torch.nn.parallel import DistributedDataParallel as DDP 
from  torch.nn.parallel import DataParallel
from .archs.restormer import Restormer
from .archs.depthrestormer import DepthRestormer
from .archs.NAFNet_arch import NAFNet
from .archs.DepthNAFNet import DepthNAFNet
#from .archs.DeblurDiNATL import NADeblurL
#from .archs.depthDeblurDiNATL import DepthNADeblurL
from .archs.stripformer import Stripformer
from .archs.depthstripformer import DepthStripformer
from .archs.catrestormer import CatRestormer
from .archs.catstripformer import CatStripformer
from .archs.catNAFNet_arch import CatNAFNet


_all__ = {
    'import_model',
    'Restormer',
    'NADeblurL',
    'Stripformer',
    'NAFNet',
    'DepthRestormer',
    'DepthStripformer',
    'DepthNAFNet',
    'DepthNADeblurL',
    'DepthNAFNet',
    'CatRestormer',
    'CatStripformer',
    'CatNAFNet',

    'import_module'
}


def import_model(opt,gpu_id=None):
    model_name = opt.config['model']['name']
    if  model_name not in _all__:
        raise ValueError('unknown model, please choose from [ Restormer, NADeblurL, Stripformer, NAFNet, DepthRestormer, DepthNAFNet, DepthNADeblurL, CatRestormer, CatStripformer, CatNAFNet ]')
    model = getattr(import_module('model'),model_name)()
    
    if opt.config['model']['testing_from_dp']:
        model = torch.nn.DataParallel(model)

    # modify this in case of errors importing pretrained models
    if opt.config['model']['resume']:
        #model.load_state_dict(torch.load(opt.config['model']['pretrained'],map_location=opt.device)['model_state_dict'],strict=True)
        ### remove prefix
        state_dict = torch.load(opt.config['model']['pretrained'],map_location=opt.device)['model_state_dict']
        state_dict = {k.partition('module.')[2]:state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict,strict=True)   
    elif opt.config['model']['pretrained']:
        #model.load_state_dict(torch.load(opt.config['model']['pretrained'],map_location=opt.device)['params'],strict=False)
        model.load_state_dict(torch.load(opt.config['model']['pretrained'],map_location=opt.device),strict=False)
        # state_dict = torch.load(opt.config['model']['pretrained'],map_location=opt.device)
        # state_dict = {k.partition('module.')[2]:state_dict[k] for k in state_dict.keys()}
        # model.load_state_dict(state_dict,strict=True)   

    model = model.to(opt.device)
    if opt.config['model']['num_gpus'] > 1:
        model =DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    return model
