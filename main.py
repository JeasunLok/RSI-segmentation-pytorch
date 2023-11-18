import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from utils.utils import *
from utils.dataloader import *
from model.UNet import *
from model.SegNext_SegModel import *
from train import *
from test import *

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
    epoch = 10
    save_period = epoch//10
    batch_size = 2

    # 学习率
    lr = 5e-3
    min_lr = lr*0.01

    # 优化器
    momentum = 0.9
    weight_decay = 0
    
    data_dir = "data"
    logs_dir = "logs"
    checkpoints_dir = "checkpoints"
    time_now = time.localtime()
    logs_folder = os.path.join(logs_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    checkpoints_folder = os.path.join(checkpoints_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    os.makedirs(logs_folder)
    os.makedirs(checkpoints_folder)

    dice_loss = False
    focal_loss = False

    print("===============================================================================")
    # 设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if Cuda:
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
    else:
        device = torch.device("cpu")
        local_rank = 0

    # model = UNet(num_classes=num_classes).to(device)
    model = SegNext_SegModel(out_channels=num_classes).to(device)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model.load_state_dict(torch.load(model_path, map_location = device))

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
    with open(os.path.join(data_dir, r"list/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(data_dir, r"list/val.txt"),"r") as f:
        val_lines = f.readlines()
    with open(os.path.join(data_dir, r"list/test.txt"),"r") as f:
        test_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    num_test = len(test_lines)

    np.random.seed(2023)

    print("device:", device, "num_train:", num_train, "num_val:", num_val, "num_test:", num_test)
    print("===============================================================================")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch//10, eta_min=min_lr) 
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
    print("start training")
    epoch_result = np.zeros([4, epoch])
    for e in range(epoch):
        model_train.train()
        train_acc, train_mIoU, train_loss = train_epoch(model_train, train_loader, criterion, optimizer, e, epoch, device, num_classes)
        scheduler.step()
        print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.2f}% | train_mIoU: {:.2f}%".format(e+1, train_loss, train_acc*100, train_mIoU*100))
        epoch_result[0][e], epoch_result[1][e], epoch_result[2][e], epoch_result[3][e]= e+1, train_loss, train_acc*100, train_mIoU*100

        if ((e+1) % save_period == 0) | (e == epoch - 1):
            print("===============================================================================")
            print("start validating")
            model_train.eval()      
            val_CM, val_acc, val_mIoU, val_loss = valid_epoch(model_train, val_loader, criterion, e, epoch, device, num_classes)
            val_weighted_recall, val_weighted_precision, val_weighted_f1 = compute_metrics(val_CM)
            if (e != epoch -1):
                print("Epoch: {:03d}  =>  Accuracy: {:.2f}% | MIoU: {:.2f}% | W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(e+1, val_acc*100, val_mIoU*100, val_weighted_recall, val_weighted_precision, val_weighted_f1))
            torch.save(model, os.path.join(checkpoints_folder, "model_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pth"))
            torch.save(model.state_dict(), os.path.join(checkpoints_folder, "model_state_dict_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pth"))
            print("===============================================================================")
    
    if distributed:
        train_sampler.set_epoch(epoch)

    if distributed:
        dist.barrier()

    draw_result_visualization(logs_folder, epoch_result)
    print("save train logs successfully")
    print("===============================================================================")

    print("start testing")
    model_train.eval()
    test_CM, test_acc, test_mIoU, val_loss = test_epoch(model_train, val_loader, device)
    test_weighted_recall, test_weighted_precision, test_weighted_f1 = compute_metrics(test_CM)
    print("Test Result  =>  Accuracy: {:.2f}%| mIoU: {:.2f} | W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(test_acc*100, test_mIoU*100, test_weighted_recall, test_weighted_precision, test_weighted_f1))
    store_result(logs_folder, test_acc, test_mIoU, test_weighted_recall, test_weighted_precision, test_weighted_f1, test_CM, epoch, batch_size, lr, weight_decay)
    print("save test result successfully")
    print("===============================================================================") 