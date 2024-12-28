import os
import math
import json
import torch
import shutil
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from model.centernet import CenterNet
from dataset.dataset import UIDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.train_utils import get_summary_image

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult    # cycle steps magnification
        self.base_max_lr = max_lr   # first max learning rate
        self.max_lr = max_lr    # max learning rate in the current cycle
        self.min_lr = min_lr    # min learning rate
        self.warmup_steps = warmup_steps    # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps    # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch     # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def remove_dir_and_create_dir(dir_name, is_remove=True):
    """
    Make new folder, if this folder exist, we will remove it and create a new folder.
    Args:
        dir_name: path of folder
        is_remove: if true, it will remove old folder and create new folder

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "create.")
    else:
        if is_remove:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            print(dir_name, "create.")
        else:
            print(dir_name, "is exist.")
            
def train_one_epoch(model, train_loader, epoch, optimizer, scheduler, device, writer, cat, input_shape):
    global step
    
    model.train()
    tbar = tqdm(train_loader)
    
    total_loss = []
    image_write_step = len(train_loader)

    for images, hms_true, whs_true, offsets_true, offset_masks_true, _ in tbar:
        tbar.set_description("epoch {}".format(epoch))

        # Set variables for training
        images = images.float().to(device)
        hms_true = hms_true.float().to(device)
        whs_true = whs_true.float().to(device)
        offsets_true = offsets_true.float().to(device)
        offset_masks_true = offset_masks_true.float().to(device)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss
        training_output = model(images, mode='train', ground_truth_data=(hms_true,
                                                                         whs_true,
                                                                         offsets_true,
                                                                         offset_masks_true))
        hms_pred, whs_pred, offsets_pred, loss, c_loss, wh_loss, off_loss, hms_true = training_output

        loss = loss.mean()
        c_loss = c_loss.mean()
        wh_loss = wh_loss.mean()
        off_loss = off_loss.mean()

        total_loss.append(loss.item())
        
        if step % image_write_step == 0:
            summary_images = get_summary_image(images, input_shape, cat, 0.5,
                                            hms_true, whs_true, offsets_true,
                                            hms_pred, whs_pred, offsets_pred, device)
            for i, summary_image in enumerate(summary_images):
                writer.add_image('train_images_{}'.format(i), summary_image, global_step=step, dataformats="HWC")

        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("c_loss", c_loss.item(), step)
        writer.add_scalar("wh_loss", wh_loss.item(), step)
        writer.add_scalar("offset_loss", off_loss.item(), step)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)

        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        tbar.set_postfix(total_loss="{:.4f}".format(loss.item()),
                         c_loss="{:.4f}".format(c_loss.item()),
                         wh_loss="{:.4f}".format(wh_loss.item()),
                         offset_loss="{:.4f}".format(off_loss.item()))

        # clear batch variables from memory
        del images, hms_true, whs_true, offsets_true, offset_masks_true

    return np.mean(total_loss)

def eval_one_epochs(model, val_loader, epoch, device, writer, cat, input_shape):

    model.eval()

    total_loss = []
    total_c_loss = []
    total_wh_loss = []
    total_offset_loss = []
    write_image = True

    with torch.no_grad():
        for images, hms_true, whs_true, offsets_true, offset_masks_true,_ in val_loader:

            # Set variables for training
            images = images.float().to(device)
            hms_true = hms_true.float().to(device)
            whs_true = whs_true.float().to(device)
            offsets_true = offsets_true.float().to(device)
            offset_masks_true = offset_masks_true.float().to(device)

            # Get model predictions, calculate loss
            training_output = model(images, mode='train', ground_truth_data=(hms_true,
                                                                             whs_true,
                                                                             offsets_true,
                                                                             offset_masks_true))
            hms_pred, whs_pred, offsets_pred, loss, c_loss, wh_loss, off_loss, hms_true = training_output

            loss = loss.mean()
            c_loss = c_loss.mean()
            wh_loss = wh_loss.mean()
            off_loss = off_loss.mean()

            total_loss.append(loss.item())
            total_c_loss.append(c_loss.item())
            total_wh_loss.append(wh_loss.item())
            total_offset_loss.append(off_loss.item())
            
            if write_image:
                write_image = False
                summary_images = get_summary_image(images, input_shape, cat, 0.5,
                                            hms_true, whs_true, offsets_true,
                                            hms_pred, whs_pred, offsets_pred, device)
                for i, summary_image in enumerate(summary_images):
                    writer.add_image('val_images_{}'.format(i), summary_image, global_step=epoch, dataformats="HWC")

            # clear batch variables from memory
            del images, hms_true, whs_true, offsets_true, offset_masks_true

        writer.add_scalar("val_loss", np.mean(total_loss), epoch)
        writer.add_scalar("val_c_loss", np.mean(total_c_loss), epoch)
        writer.add_scalar("val_wh_loss", np.mean(total_wh_loss), epoch)
        writer.add_scalar("val_offset_loss", np.mean(total_offset_loss), epoch)

    return np.mean(total_loss)

def main():
    # First create ./log dict to store the results
    # By default we will name the folder according to the current time
    now =  datetime.now()
    weight_path = "./logs/{}-{}-{}-{}-{}-{}/level/weights".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    summary_path = "./logs/{}-{}-{}-{}-{}-{}/level/summary".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    remove_dir_and_create_dir(weight_path)
    remove_dir_and_create_dir(summary_path)
    
    # run on gpu if cuda exists else cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on cuda")
    else:
        device = torch.device("cpu")
        print("Running on cpu")
        
    # hyper params
    global step
    step = 0
    input_shape = (1280, 1920)  # Please ensure the number you've put here can be devided by 32
    batch_size = 1
    init_lr = 5e-3
    end_lr = 1e-5
    freeze_epoch = 0
    unfreeze_epoch = 100
    target = "level"
    data_path = "./data"
    
    # write hyper params to .json file
    hyper_params = {
        "input_shape": input_shape,
        "batch_size": batch_size,
        "init_lr": init_lr,
        "end_lr": end_lr,
        "freeze_epoch": freeze_epoch,
        "unfreeze_epoch": unfreeze_epoch,
        "target": target
    }
    with open("{}/hyper_params.json".format(summary_path), 'w') as f:
        json.dump(hyper_params, f, indent=4)

    # get CenterNet model
    num_cat = 4 if target == "class" else 3
    category =["clickable", "selectable", "scrollable", "disabled"] if target == "class" else ["level_0", "level_1", "level_2"]
    model = CenterNet(backbone="resnet101", num_classes=num_cat)
    model.to(device)
    print("Model create successful.")
    
    # get train test dataset
    train_dataset = UIDataset(data_path, input_shape=input_shape, is_train=True, target=target)  
    test_dataset = UIDataset(data_path, input_shape=input_shape, is_train=False, target=target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    writer = SummaryWriter(summary_path)
    
    freeze_step = len(train_dataset) // batch_size
    unfreeze_step = len(train_dataset) // batch_size
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, init_lr)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=freeze_epoch*freeze_step + unfreeze_epoch*unfreeze_step,
                                              max_lr=init_lr, 
                                              min_lr=end_lr)
    
    # freeze 
    if freeze_epoch > 0:
        print("Freeze backbone and decoder, train {} epochs.".format(freeze_epoch))
        model.freeze_backbone()
        for epoch in range(freeze_epoch):
            train_loss = train_one_epoch(model, train_loader, epoch, optimizer, scheduler, device, writer, category, input_shape)
            val_loss = eval_one_epochs(model, test_loader, epoch, device, writer, category, input_shape)
            print("=> loss: {:.4f}   val_loss: {:.4f}".format(train_loss, val_loss))
            torch.save(model,
                       '{}/epoch={}_loss={:.4f}_val_loss={:.4f}.pt'.
                       format(weight_path, epoch, train_loss, val_loss))
    
    # unfreeze
    if unfreeze_epoch > 0:
        print("Unfreeze backbone and decoder, train {} epochs.".format(unfreeze_epoch))
        model.unfreeze_backbone()
        for epoch in range(unfreeze_epoch):
            epoch += freeze_epoch
            train_loss = train_one_epoch(model, train_loader, epoch, optimizer, scheduler, device, writer, category, input_shape)
            val_loss = eval_one_epochs(model, test_loader, epoch, device, writer, category, input_shape)
            print("=> loss: {:.4f}   val_loss: {:.4f}".format(train_loss, val_loss))
            torch.save(model,
                       '{}/epoch={}_loss={:.4f}_val_loss={:.4f}.pt'.
                       format(weight_path, epoch, train_loss, val_loss))

if __name__ == "__main__":
    main()