import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np
from pcdet.models import load_data_to_gpu
import copy
import pcdet.datasets.augmentor.augmentor_utils as uti


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    # reset DataLoader Iterator if total number of interactions matches dataset length 
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    # initialize the progress bar only for GPU 0
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    accus = 1

    # Training loop over mini-batches
    for cur_it in range(total_it_each_epoch): # iterate over the total number of iterations in one epoch
        
        # Fetch the next training batch
        try: 
            batch = next(dataloader_iter) 
        except StopIteration: # if the iterator has reached the end of a dataset, reset the iterator and fetch the next batch
            dataloader_iter = iter(train_loader) # reset the iterator
            batch = next(dataloader_iter)
            print('new iters')

        # Update learning rate dynamically based on the current iteration
        lr_scheduler.step(accumulated_iter)

        try: # Retrieve current learning rate from optimizer
            cur_lr = float(optimizer.lr) 
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None: # log the learning rate to tensorboard
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        # Train the model - Forward Pass
        model.train() #set model to training mode
    
        loss, tb_dict, disp_dict = model_func(model, batch) #runs a forward pass to compute loss
        loss = loss/accus # divide the loss by the number of accumulation steps (here = 1)
        
        # Backpropagation
        loss.backward()

        # Gradient clipping to prevent exploding gradients and Optimizer update 
        if ((cur_it + 1) % accus) == 0:
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step() # update model parameters 
            optimizer.zero_grad() # reset gradients to prevent accumulation
        
        # increment the iteration counter to keep track of the total number of iterations
        accumulated_iter += 1
        disp_dict.update({'loss': loss.item()*accus, 'lr': cur_lr}) # update the progress bar display with the current loss and learning rate



        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    
    # initialize the iteration counter depending on the starting epoch and iteration (e.g. from args)
    accumulated_iter = start_iter
    
    # tqdm.trange used for a better visulization in the console
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader) # calculates number of iterations per epoch
        if merge_all_iters_to_one_epoch: 
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader) # creates an iterator for the train_loader to manually fetch next batch with next(dataloader_iter) instead of using for loop
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            ###### Train one epoch #####
            if lr_warmup_scheduler is not None:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                # save checkpoint
                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                
                
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename,_use_new_zipfile_serialization=False)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename,_use_new_zipfile_serialization=False)
