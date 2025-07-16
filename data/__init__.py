from torch.utils import data
from importlib import import_module
#from .deblurdata import DeblurData
#from .depthdeblurdata import DepthDeblurData
from .kerneldepthdeblur import KernelDepthDeblurData
from .kerneldeblurdata import KernelDeblurData
__all__ = {
    'DeblurData',
    'DepthDeblurData',
    'KernelDepthDeblurData',
    'kernelDeblurData'
    'import_loader'
}


def import_loader(opt,num_replicas=None, rank =None):
    dataset_name = opt.model_task + 'Data'
    dataset = getattr(import_module('data'), dataset_name)

    if opt.task == 'train':
        is_patch = False
        train_inp_path = opt.config['train']['train_inp']
        valid_inp_path = opt.config['train']['valid_inp']
        valid_gt_path = opt.config['train']['valid_gt']
        if opt.model_task == 'DepthDeblur':
            train_gt_path = opt.config['train']['train_gt']
            train_dep_path = opt.config['train']['train_dep']
            valid_dep_path = opt.config['train']['valid_dep']
            if opt.config['train']['patch_size'] >0:
                is_patch = True
            train_data = dataset(opt, train_inp_path, train_gt_path, train_dep_path, is_patch)
            valid_data = dataset(opt, valid_inp_path, valid_gt_path, valid_dep_path,is_patch=False)
        elif opt.model_task == 'KernelDepthDeblur':
            train_dep_path = opt.config['train']['train_dep']
            valid_dep_path = opt.config['train']['valid_dep']
            if opt.config['train']['patch_size'] >0:
                is_patch = True
            train_data = dataset(opt, train_inp_path, False, train_dep_path, is_patch)
            valid_data = dataset(opt, valid_inp_path, valid_gt_path, valid_dep_path,is_patch=False)            
        elif opt.model_task == "KernelDeblur":
            if opt.config['train']['patch_size'] >0:
                is_patch = True
            train_data = dataset(opt, train_inp_path, False, is_patch)
            valid_data = dataset(opt, valid_inp_path, valid_gt_path,False)
        elif opt.model_task == "Deblur":
            train_gt_path = opt.config['train']['train_gt']
            if opt.config['train']['patch_size'] >0:
                is_patch = True
            train_data = dataset(opt, train_inp_path, train_gt_path, is_patch)
            valid_data = dataset(opt, valid_inp_path, valid_gt_path,False)
    
        if num_replicas !=None and rank !=None:
            train_sampler = data.distributed.DistributedSampler(
            train_data, num_replicas=num_replicas, rank=rank)
            # vaild_sampler = data.distributed.DistributedSampler(
            # valid_data,num_replicas=num_replicas, rank=rank)
            train_loader = data.DataLoader(
            train_data,
            batch_size=opt.config['train']['batch_size'],
            num_workers=opt.config['train']['num_workers'],
            sampler = train_sampler,
            drop_last=True,
            )

        else:
            train_loader = data.DataLoader(
                train_data,
                batch_size=opt.config['train']['batch_size'],
                shuffle=True,
                num_workers=opt.config['train']['num_workers'],
                drop_last=True,)
        valid_loader = data.DataLoader(
                valid_data,
                batch_size=1,
                shuffle=False,
                num_workers=opt.config['train']['num_workers'],
                drop_last=False,    
         )

        return train_loader, valid_loader

    elif opt.task == 'test':
        inp_test_path = opt.config['test']['test_inp']
        gt_test_path = opt.config['test']['test_gt']
        if opt.model_task == 'DepthDeblur' or opt.model_task == "KernelDepthDeblur":
            dep_test_path = opt.config['test']['test_dep']
            test_data = dataset(opt, inp_test_path, gt_test_path, dep_test_path)
        else:
            test_data = dataset(opt, inp_test_path, gt_test_path)
        test_loader = data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['test']['num_workers'],
            drop_last=False,
        )
        return test_loader


    else:
        raise ValueError('unknown task, please choose from [train, test]')
