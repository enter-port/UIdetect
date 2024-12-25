'''
Training process of the dino
Get inspiration mainly from engine.py in DINO
ADD tensorboard and visualization in the process of training
Delete contents of distributed GPU training
'''
# add the model path of DINO into current sys path
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./third_party/DINO_UI/models/dino"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./third_party/DINO_UI"))
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Iterable 
import math
import random
from datetime import datetime
import time
from collections import OrderedDict
import json
import pickle

from dataset.dino_dataset import UIDataset
from torch.utils.data import DataLoader

from third_party.DINO_UI.models.dino.dino import build_dino
from utils.arg_utils import create_dino_args
from utils.train_utils import get_param_dict, create_directories, save_model, visualize_and_save, deprocess_input
import utils.misc as utils

from tensorboardX import SummaryWriter

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    train_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, 
                    args=None, writer=None, ema_m=None):
    global step
    
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
        
    if args is None:
        raise ValueError("args must not be None")
    if writer is None:
        raise ValueError("writer must not be None")

    model.train()
    criterion.train()
    tbar = tqdm(train_loader)
    
    # initialize loss_stats and loss_dict
    loss_stats = {}
    loss_values = []
    
    for samples, targets in tbar:

        samples = samples.float().to(device)
        # targets is a list a dict: (batch_size * len(dict))
        targets = [{k: v[i].float().to(device) for k, v in targets.items()} for i in range(args.batch_size)]
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # used amp in training process to accelerate training
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
                
            # calculate loss by criterion 
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes, here we just omit distributed training
        loss_value = losses.item() # loss_value is a float

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        loss_values.append(loss_value)

        # original backward function, no amp used in backward 
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # different param updating strategies based on arguments
        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)
        
        # Update the loss stats after each batch
        for k, v in loss_dict.items():
            if k not in loss_stats:
                loss_stats[k] = []
            loss_stats[k].append(v.item())
            
        # Log scalar values to TensorBoard for the overall loss
        if writer is not None:
            writer.add_scalar("loss", loss_value, step)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)
            # log averaged individual losses from loss_stats to TensorBoard
            for loss_name, loss_value_item in loss_stats.items():
                # Log each individual loss (e.g., classification loss, bbox loss, etc.)
                writer.add_scalar(f"{loss_name}_loss", np.mean(loss_value_item), step)

        # Update the progress bar with overall loss in one batch
        tbar.set_postfix(loss="{:.4f}".format(loss_value))
        
        # Increment the step counter
        step += 1
            
    # Calculate the mean of each loss over the epoch
    for k, v in loss_stats.items():
        loss_stats[k] = sum(v) / len(v)

    # following are two schedulars for criterions (loss func)
    # update criterion weight dict if changed

    if args.finetune:
        if getattr(criterion, 'loss_weight_decay', False):
            criterion.loss_weight_decay(epoch=epoch)
        if getattr(criterion, 'tuning_matching', False):
            criterion.tuning_matching(epoch)
    
    return loss_stats, np.mean(loss_values)

@torch.no_grad()
def eval_one_epoch(model, criterion, postprocessors, eval_loader, 
                   device, current_epoch, 
                   wo_class_error=False, args=None, writer=None):
    
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    
    if args is None:
        raise ValueError("args must not be None")
    if writer is None:
        raise ValueError("writer must not be None")
    
    model.eval()
    criterion.eval()
    
    # here, the eval of our result is based on the given script
    # so igonre the initialization with cocoevaluator
    
    tbar = tqdm(eval_loader)
    loss_stats = {}
    loss_values = []
    output_state_dict = {}
    
    for samples, targets in tbar:
        samples = samples.float().to(device)
        # targets is a list of dict
        # targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        targets = [{k: v[i].float().to(device) for k, v in targets.items()} for i in range(args.batch_size)]
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes, here we just omit distributed training
        loss_value = losses.item() # loss_value is a float
        loss_values.append(loss_value)
        
        # post_process procedure
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes) # [scores: [num_of_boxes], labels: [num_of_boxes], boxes: [num_of_boxes, 4]] x B
        
        # no mask used in this task, so "segm" can be ignored
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        # put the result in standarized format if needed
        for i, (tgt, res) in enumerate(zip(targets, results)):
            """
            pred vars:
                K: number of bbox pred
                score: Tensor(K),
                label: list(len: K),
                bbox: Tensor(K, 4)
                idx: list(len: K)
            tgt: dict.
            经过后处理的result['boxes'],代表每个图片中每个框的坐标 [x_min, y_min, x_max, y_max]，且这个坐标是经过反归一化的原始坐标
            """
            # compare gt and res (after postprocess)
            gt_bbox = tgt['boxes']
            gt_label = tgt['labels']
            gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
        
            _res_bbox = res['boxes']
            _res_prob = res['scores']
            _res_label = res['labels']
            res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
            # import ipdb;ipdb.set_trace()

            if 'gt_info' not in output_state_dict:
                output_state_dict['gt_info'] = []
            output_state_dict['gt_info'].append(gt_info.cpu())

            if 'res_info' not in output_state_dict:
                output_state_dict['res_info'] = []
            output_state_dict['res_info'].append(res_info.cpu())
                
        # Update the loss stats after each batch
        for k, v in loss_dict.items():
            if k not in loss_stats:
                loss_stats[k] = []
            loss_stats[k].append(v.item())
            
        # Update the progress bar with overall loss in one batch
        tbar.set_postfix(loss="{:.4f}".format(loss_value))
        
    # Calculate the mean of each loss over the epoch
    for k, v in loss_stats.items():
        loss_stats[k] = sum(v) / len(v)     
        
    # Log scalar values to TensorBoard for the overall loss after each epoch
    if writer is not None:
        writer.add_scalar("average loss after this epoch", np.mean(loss_values), current_epoch)
        # Also log individual losses from loss_stats to TensorBoard
        for loss_name, loss_value_item in loss_stats.items():
            # Log each individual loss (e.g., classification loss, bbox loss, etc.)
            writer.add_scalar(f"{loss_name}_loss_after_epoch_{current_epoch}", np.mean(loss_value_item), current_epoch)
                
    return loss_stats, np.mean(loss_values), output_state_dict

def main():
    now =  datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    # make directories to save models and ouputs in the process of training
    base_log_path = f"./logs/{timestamp}"
    subdirs = ['weights', 'summary', 'result', 'vis']
    weight_path, summary_path, result_path, vis_path = create_directories(base_log_path, subdirs)

    # obtain the numbers of categories from .txt file
    category_path = "./data/categories.txt"
    category = []
    with open(category_path, 'r', encoding='utf-8') as file:
        category = [line.strip() for line in file]
    num_cat = len(category)
    
    # set params for training
    args = create_dino_args()
    # get the current usable device
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # craete model, loss function and postprocessor
    model, criterion, postprocessors = build_dino(args)  # DINO_4scale
    model.to(device)
    
    # set optimizer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_dicts = get_param_dict(args, model)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # lr warm-up, used for fine-tuning
    def lr_lambda(step):
        warmup_steps = 400
        if step < warmup_steps:
            return float(step) / warmup_steps
        return 1.0  
    
    lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # set lr_schedular, the alternative one includes lr warmup
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # set tensorboard
    writer = SummaryWriter(summary_path)
        
    # load model from given path
    if args.pre_train and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
                
        # freeze some weights and ignore some weights when fine-tunning
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        _frozenwordlist = args.frozen_weights if args.frozen_weights else []
        
        # freeze the ignored parameters
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in _frozenwordlist):
                param.requires_grad = False 
        
        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    # ignorelist.append(keyname)
                    return False
            return True
        
        # _tmp_st is the keys not frozen in this process
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        
        # load the keys not frozen to the model
        # Here use strict = False since we ignore(frozen) some of the parameters for better fine-tuning
        _load_output = model.load_state_dict(_tmp_st, strict=False)
        print(str(_load_output))
    
    global step
    step = 0

    # Here, due to lack of training data, we don't sample on the original dataset
    train_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=args.input_shape, is_train=True)
    test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=args.input_shape, is_train=False) 
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    
    # 对现有模型进行测试
    if args.eval:
        print("Evaluation the existing model!")
        # evaluation of this process
        os.environ['EVAL_FLAG'] = 'TRUE'
        eval_stats, average_eval_loss, result_dict = eval_one_epoch(
            model, criterion, postprocessors, eval_loader, 
            device, current_epoch = 1, wo_class_error=args.wo_class_error, args=args, writer=writer
        )
        
        # Evaluation process provided by TA!
        
        # visualization process on first several images
        
        print(f'Average evaluation loss of this model:{average_eval_loss}')
        return

    print("Start training")
    start_time = time.time()
    # 训练及测试框架
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # training process
        train_stats, averge_train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=args.wo_class_error, lr_scheduler=lr_scheduler, args=args, writer=writer, ema_m=None
        )
        
        if not args.onecyclelr:
            if args.finetune:
                lr_scheduler_warmup.step()
            lr_scheduler.step()
            
        # evaluation process
        eval_stats, average_eval_loss, result_dict = eval_one_epoch(
            model, criterion, postprocessors, eval_loader, 
            device, epoch, wo_class_error=args.wo_class_error, args=args, writer=writer
        )
        
        # Evaluation process! based on TA's script
        # Maybe use result dict here for evaluation
        
        # visualization of the first graph
        if args.vis:
            first_image_gt_info = result_dict['gt_info'][0]
            first_image_res_info = result_dict['res_info'][0]
            # print("gt info of the first image:", first_image_gt_info)
            # print("predicted info of the first image:", first_image_res_info)
            image, target = test_dataset[0] # image: (3,1280, 1960), np.ndarray
            image_restored = deprocess_input(image)
            original_shape = (1280, 1960)
            image_name = f'image_{epoch}'
            visualize_and_save(image_restored, first_image_gt_info, first_image_res_info, save_dir=vis_path, image_name=image_name, original_size=original_shape)
        
        print("=> train loss: {:.4f}   val loss: {:.4f}".format(averge_train_loss, average_eval_loss))
            
        # save model after every epoch
        # After we have evaluation process, we can save model only when better result is obtained
        # Currently we simply save model after each epoch
        save_model(model, epoch, weight_path, optimizer)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_duration:.2f}s")
        
        # save the result dict
        file_name = f'res_dict_{epoch}.pkl'
        file_path = os.path.join(result_path, file_name)
        if args.save_results:
            with open(file_path, 'wb') as f:
                pickle.dump(result_dict, f)
            print(f"res_dict has been saved to {file_path}")
        
    total_duration = time.time() - start_time
    print(f"Training completed in {total_duration:.2f}s")    

if __name__ == '__main__':
    main()
 