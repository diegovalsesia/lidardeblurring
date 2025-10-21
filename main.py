import torch
import numpy as np
import cv2
from tqdm import tqdm
import os
from logger import Logger
from option import get_option
from data import import_loader
from loss import import_loss
from model import import_model
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter 
from img_util import calculate_ssim_pt
import lpips
def train(gpu_id, opt, logger):
    logger = Logger(opt)
    if gpu_id ==0:
        writer = SummaryWriter(opt.tensorboard_dir)
    if  opt.config['model']['num_gpus']>1:
        rank = gpu_id
        world_size =  opt.config['model']['num_gpus'] * opt.config['model']['node']
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(gpu_id)
        train_loader, valid_loader = import_loader(opt,num_replicas=world_size, rank = rank)
    else:
        train_loader, valid_loader = import_loader(opt)
    if gpu_id ==0:
        logger.info('task: {}, model task: {}'.format(opt.task, opt.model_task))
    #train_loader, valid_loader = import_loader(opt)
    lr = float(opt.config['train']['lr'])
    loss_training = import_loss(opt.model_task, opt)
    net = import_model(opt,gpu_id)
    """
    Following Code is added for prompt tuning
    """
    if opt.config['train']['prompt_tuning']:
        for name, para in net.named_parameters():
            if 'adapter' in name:
                para.required_grad = True
                if gpu_id == 0:
                    logger.info('the parameters {} is trainable'.format(name))
            else:
                para.requires_grad = False

    if opt.config['train']['encoder_freezing']:
         for name, para in net.named_parameters():
             if 'encoder' or 'down' in name:
                para.requires_grad = False
    net.train()
    # logger.info(net)
    # Phase Warming-up
    # if opt.config['train']['warmup']:
    #     logger.info('start warming-up')

    #     optim_warm = torch.optim.Adam(net.parameters(), lr_warmup, weight_decay=0)
    #     epochs = opt.config['train']['warmup_epoch']
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #     # for epo in range(epochs):
    #     #     loss_li = []
    #     #     for img_inp, img img_gt, _ in tqdm(train_loader, ncols=80):
    #     #         optim_warm.zero_grad()
    #     #         warmup_out1, warmup_out2 = net.forward_warm(img_inp)
    #     #         loss = loss_warmup(img_inp, img_gt, warmup_out1, warmup_out2)
    #     #         loss.backward()
    #     #         optim_warm.step()
    #     #         loss_li.append(loss.item())

    #     #     logger.info('epoch: {}, train_loss: {}'.format(epo+1, sum(loss_li)/len(loss_li)))
    #     #     torch.save(net.state_dict(), r'{}\model_pre.pkl'.format(opt.save_model_dir))
    #     # logger.info('warming-up phase done')

    # Phase Training
    best_psnr = 0
    epochs = int(opt.config['train']['epoch'])
    optim = torch.optim.Adam(net.parameters(), lr, weight_decay=0)
    #lr_sch = torch.optim.lr_scheduler.MultiStepLR(optim, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],0.5)
    if opt.config['model']['resume']:
        optim.load_state_dict(torch.load(opt.config['model']['pretrained'])['optimizer_state_dict'])
        start_epoch = torch.load(opt.config['model']['pretrained'])['epoch']
        lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 2, 1e-7,last_epoch=start_epoch+1)
    else:
        lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 2, 1e-7)
        start_epoch = -1
    if gpu_id ==0:
        logger.info('start training')
    for epo in range(start_epoch+1, epochs):
        loss_li = []
        test_psnr = []
        net.train()
        if opt.model_task == 'DepthDeblur' or opt.model_task == "KernelDepthDeblur" or opt.model_task == "KernelDepthEstDeblur":
            for img_inp,img_dep, img_gt, _ in tqdm(train_loader, ncols=80): 
                out = net(img_inp, img_dep)
                loss = loss_training(out, img_gt)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_li.append(loss.item())
                if gpu_id ==0:
                    writer.add_scalar('loss', loss.item(),epo)
            lr_sch.step()
        elif opt.model_task == 'Deblur' or opt.model_task == "KernelDeblur":
            for img_inp, img_gt, _ in tqdm(train_loader, ncols=80): 
                out = net(img_inp)
                loss = loss_training(out, img_gt)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_li.append(loss.item())
                if gpu_id ==0:
                    writer.add_scalar('loss', loss.item(),epo)
            lr_sch.step()
        else:
             raise ValueError('unknown task, please choose from [DepthDeblur, Deblur, KernelDeblur, KernelDepthDeblur]')
        # Validation
        dist.barrier()
        net.eval()
        if opt.model_task == 'DepthDeblur' or opt.model_task == "KernelDepthDeblur" or opt.model_task == "KernelDepthEstDeblur":
            for img_inp, img_dep, img_gt, _ in tqdm(valid_loader, ncols=80):
                with torch.no_grad():
                    out = net(img_inp,img_dep)
                    mse = ((out - img_gt)**2).mean((2, 3))
                    psnr = (1 / mse).log10().mean() * 10
                test_psnr.append(psnr.item())
        elif opt.model_task == 'Deblur' or opt.model_task == "KernelDeblur":
            for img_inp, img_gt, _ in tqdm(valid_loader, ncols=80):
                with torch.no_grad():
                    out = net(img_inp)
                    mse = ((out - img_gt)**2).mean((2, 3))
                    psnr = (1 / mse).log10().mean() * 10
                test_psnr.append(psnr.item())
        else:
            raise ValueError('unknown task, please choose from [DepthDeblur, Deblur]')
        dist.barrier()
        mean_psnr = sum(test_psnr)/len(test_psnr)
        if gpu_id ==0:
            writer.add_scalar('psnr', mean_psnr,epo)
        if (epo+1) % int(opt.config['train']['save_every']) == 0:
            # torch.save(net.state_dict(), r'{}/model_{}.pkl'.format(opt.save_model_dir, epo+1))
            if gpu_id ==0:
                torch.save({
                'epoch': epo,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss_li,
                }, r'{}/model_{}.pth'.format(opt.save_model_dir, epo+1))
                # logger.info('epoch: {}, training loss: {}, validation psnr: {}'.format(
                #     epo+1, sum(loss_li) / len(loss_li), sum(test_psnr) / len(test_psnr)
                # ))
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            if gpu_id == 0:
                # torch.save(net.state_dict(), r'{}/model_best.pkl'.format(opt.save_model_dir))
                torch.save({
                'epoch': epo,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss_li,
                }, r'{}/model_best.pth'.format(opt.save_model_dir))
        if gpu_id == 0:
            logger.info('epoch: {}, training loss: {}, validation psnr: {}'.format(
                epo+1, sum(loss_li) / len(loss_li), sum(test_psnr) / len(test_psnr)
            ))
    if gpu_id == 0:
        logger.info('training done')


def test(opt, logger):
    test_loader = import_loader(opt)
    net = import_model(opt)
    net.eval()
    psnr_list = []
    ssim_list = []
    logger.info('start testing')
    loss_fn_alex = lpips.LPIPS(net='alex').to(opt.device)
    lpips_list = []

    if opt.model_task == 'Deblur' or opt.model_task == "KernelDeblur":
        for (img_inp, img_gt, img_name) in test_loader:

            with torch.no_grad():
                out = net(img_inp)
                mse = ((out - img_gt)**2).mean((2, 3))
                psnr = (1 / mse).log10().mean() * 10
                ssim = calculate_ssim_pt(out,img_gt,crop_border=0)
                lpips_val = loss_fn_alex(out,img_gt)

            if opt.config['test']['save']:
                out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
                cv2.imwrite(r'{}/{}.png'.format(opt.save_image_dir, img_name[0]), out_img)

            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            lpips_list.append(lpips_val.item())
            logger.info('image name: {}, test psnr: {}, test ssim:{}, test lpips:{}'.format(img_name[0], psnr, ssim.item(),lpips_val.item()))       
    elif opt.model_task =='DepthDeblur'  or opt.model_task == "KernelDepthDeblur" or opt.model_task == "KernelDepthEstDeblur":
         for (img_inp, img_dep, img_gt, img_name) in test_loader:
    
            with torch.no_grad():
                out = net(img_inp, img_dep)
                mse = ((out - img_gt)**2).mean((2, 3))
                psnr = (1 / mse).log10().mean() * 10
                ssim = calculate_ssim_pt(out,img_gt,crop_border=0)
                lpips_val = loss_fn_alex(out,img_gt)


            if opt.config['test']['save']:
                out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
                cv2.imwrite(r'{}/{}.png'.format(opt.save_image_dir, img_name[0]), out_img)

            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            lpips_list.append(lpips_val.item())
            logger.info('image name: {}, test psnr: {}, test ssim:{},test lpips:{}'.format(img_name[0], psnr, ssim.item(),lpips_val.item()))       

    logger.info('testing done, overall psnr: {}, overall ssim:{}, overall lpips:{}'.format(sum(psnr_list) / len(psnr_list),sum(ssim_list) / len(ssim_list),sum(lpips_list)/len(lpips_list)))


if __name__ == "__main__":
    opt = get_option()
    logger = Logger(opt)
    if opt.config['model']['num_gpus'] > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "8010"
        if opt.config['model']['gpu_ids'] != False:
            os.environ['CUDA_VISIBLE_DEVICES']= opt.config['model']['gpu_ids']
    if opt.task == 'train':
        if opt.config['model']['num_gpus'] > 1:
            torch.multiprocessing.set_start_method("spawn")
            torch.multiprocessing.spawn(train, args=(opt,logger), nprocs=opt.config['model']['num_gpus'], join=True)
        else:
            train(0, opt,logger)
    elif opt.task == 'test':
        test(opt, logger)
    else:
        raise ValueError('unknown task, please choose from [train, test].')
