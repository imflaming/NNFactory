from logger import logger
from config import config
import torch

from torch.utils.data import DataLoader
from data_process.scene_classify import SenceDataset

from others.lbs import LSR
from transforms.image_prepare import process_image

from torch import nn
from parts.bodys import Resnet
from parts.heads import SenceClassifyHead

from loss_function.classify_loss import one_hot_cross_entropy_loss
import numpy as np
import random

import torch.optim as optim
from lr_scheduler.gradual_warmup import GradualWarmupScheduler
from lr_scheduler.status_lr import PhaseReductionScheduler 
from utils.plot import plot
from utils.train_prepare import setup, cleanup
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from train_step.classify_train_step import classify_step_train
import os
import time
from data_process.kaggle_CMI_prepare import read_parquet_file,read_data_dictionary , processor_train_data_set

seed = 42  # 可以换成你想要的种子值
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 如果使用 CUDA
torch.cuda.manual_seed_all(seed)  # 如果使用多卡
np.random.seed(seed)
random.seed(seed)

class Test(nn.Module):
    def __init__(self, num_classes: int ) -> None:
        super().__init__()
        self.backbone = Resnet('18')
        self.head = SenceClassifyHead(num_classes)

    def forward(self, input ):
        logger.debug(f"input.shape : {input.shape} ")
        output = self.backbone( input )
        output = self.head( output )
        logger.debug(f"output.shape : {output.shape} ")
        return output

def train(rank, world_size):
    logger.debug(f"start train  rank{ rank }  world_size {world_size}")
    # 使用 DistributedSampler 确保每个进程处理不同的数据
    
    logger.debug(f"init model init")
    model = Test(config.CLASSES_NUM)    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank] ,output_device=rank )
    logger.debug(f"init model over")
    device = f"cuda:{rank}"
    logger.debug(f"device : {device}")


    lsr = LSR()
    label_processor = lsr._one_hot

    dataset = SenceDataset(data_path=config.DATA_PATH ,
                        num_class=config.CLASSES_NUM,
                        label_process= label_processor,
                        image_process= process_image)
    
    val_dataset = SenceDataset (data_path=config.DATA_PATH ,
                        num_class=config.CLASSES_NUM,
                        label_process= label_processor,
                        image_process= process_image)
    
    logger.debug(f"init train dataset over")
    train_sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, config.TRAIN_BATCH_SIZE, sampler=train_sampler,shuffle=False)
    logger.debug(f"init train dataloader over")

    logger.debug(f"init val dataset over")
    val_sampler = DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, 1, sampler=val_sampler,shuffle=False)
    logger.debug(f"init val dataloader over")


    optimizer = optim.Adam(model.parameters(), lr=config.MAX_TRAIN_LR)
    phase = config.LR_PHASE
    after_scheduler = PhaseReductionScheduler(optimizer ,phase , total_epochs=config.TOTAL_EPOCHS, start_epoch = config.WARMUP_EPOCHS)
    scheduler = GradualWarmupScheduler(optimizer, start_warmup_lr = config.MIN_TRAIN_LR, total_epoch = config.WARMUP_EPOCHS, after_scheduler=after_scheduler)
    
    lr_list = []
    step_count_per_epoch = len(dataloader)
    epoch_loss_list = []
    for epoch in range(config.TOTAL_EPOCHS):
        epoch_loss = 0
        model.train()
        train_sampler.set_epoch(epoch)  # 每个epoch需要设置新的 epoch
        
        
        for idx, ( input , label ) in  enumerate(dataloader):
            optimizer.zero_grad()
            
            input = input.to(device)
            label = label.to(device)

            loss = classify_step_train(model, input, label, loss_function=one_hot_cross_entropy_loss ,optimizer = optimizer)
            epoch_loss += loss
            loss.backward() # 反向传播
            optimizer.step() # 
            logger.debug(f"rank {rank} one step over ")

        scheduler.step()
        epoch_lr = optimizer.param_groups[0]['lr']
        lr_list.append(epoch_lr)
        epoch_loss_list.append(epoch_loss.cpu().detach().numpy())

        logger.debug(f"rank {rank} epoch : {epoch} scheduler.lr { epoch_lr }")
        logger.info(f"rank {rank} epoch : {epoch} loss = {epoch_loss} mean_loss = {epoch_loss/step_count_per_epoch} epoch_lr = {epoch_lr}")

        model.eval()

        #======================================================================================================================#
        right = 0
        start_t = time.time()
        val_count = len(train_dataloader)
        for idx, ( input , label ) in  enumerate(train_dataloader):
            input = input.to(device)
            label = label.to(device)
        
            pred = model(input)
            max_values, max_indices = torch.max(pred, dim=1)
            _, label = torch.max(label, dim=1)
            if max_indices == label :
                right +=1
        end_t = time.time()
        logger.info(f"rank {rank} epoch : {epoch} eval acc = {right/val_count} eval time = {(end_t-start_t)/val_count} ")


    epoch_x = list(range(config.TOTAL_EPOCHS))

    if rank == 0:
        # 绘制学习率变化曲线
        plot(epoch_x,[lr_list] ,y_name_list= ['lr'],table_name= "lr scheduler" , xlabel='epoch',save_path = f"test_{rank}.jpg")    
        plot(epoch_x,[epoch_loss_list], y_name_list = ['train_loss'], table_name = "train_loss" ,xlabel='epoch', save_path=f"train_loss_{rank}.jpg")

    logger.debug(f"model body : {model.module.backbone}")
    cleanup()

class Trainner:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def train(self):
        logger.debug(f"train a model start ... rank{self.rank+1}/{self.world_size}")

        import os
        root = "D:\dataset\CMI"
        dicitionary = "data_dictionary.csv"
        file_path = r"D:\dataset\CMI\series_test.parquet\id=001f3379\part-0.parquet"
        processor_train_data_set(os.path.join(root, "train.csv"))

        pass


