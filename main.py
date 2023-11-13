import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.utils import *
from utils.dataloader import *
from model.UNet import *
from train import *

if __name__ == "__main__":
    # 是否使用GPU
    Cuda = False
    num_workers = 4
    distributed = False
    sync_bn = False
    fp16 = False

    num_classes = 11

    pretrained = False
    model_path = r""

    input_shape = [512, 512]
    epoch = 100
    batch_size = 2
    # Freeze_Train = True

    # 学习率
    lr = 7e-3

    # 优化器
    momentum = 0.9
    weight_decay = 0
    
    data_path = "data"
    save_dir = "logs"
    dice_loss = False
    focal_loss = False

    # 设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    model = UNet(num_classes=num_classes).to(device)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model.load_state_dict(torch.load(model_path, map_location = device))
    
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    else:
        loss_history = None

    # 混精度
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(data_path, r"list/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(data_path, r"list/val.txt"),"r") as f:
        val_lines = f.readlines()
    with open(os.path.join(data_path, r"list/test.txt"),"r") as f:
        test_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    num_test = len(test_lines)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//10, gamma=0.9) 
    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

    image_transform = get_transform(input_shape, IsTotensor=True, IsNormalize=True)
    label_transform = get_transform(input_shape, IsTotensor=True, IsNormalize=False)
    # image_transform = None
    # label_transform = None
    train_dataset = MyDataset(train_lines, input_shape, num_classes, image_transform=image_transform, label_transform=None)
    val_dataset = MyDataset(val_lines, input_shape, num_classes, image_transform=image_transform, label_transform=None)
    test_dataset = MyDataset(test_lines, input_shape, num_classes, image_transform=image_transform, label_transform=None)

    if distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        test_sampler     = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False,)
        batch_size      = batch_size // ngpus_per_node
        shuffle         = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True,  sampler=train_sampler)
    val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=test_sampler)

    # 开始模型训练
    for e in range(epoch):
        train_acc1, train_mIoU, train_loss = train_epoch(model_train, train_loader, criterion, optimizer, e, epoch, device, num_classes)
        scheduler.step()
        # train()
        # val()
        # test()


        # if distributed:
        #     train_sampler.set_epoch(epoch)

        # if distributed:
        #     dist.barrier()

    