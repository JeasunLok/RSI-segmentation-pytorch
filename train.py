import numpy as np
import torch
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.accuracy import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, e, epoch, device, num_classes):
    loss_show = AverageMeter()
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.to(device).float()
        batch_label = batch_label.to(device).long()

        optimizer.zero_grad()
        batch_prediction = model(batch_data)
        loss = criterion(batch_prediction, batch_label)
        loss.backward()
        optimizer.step()       

        batch_prediction = F.softmax(batch_prediction, dim=1)
        batch_prediction = torch.argmax(batch_prediction, dim=1)
        # calculate the accuracy
        CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
        CM = CM + CM_batch
        n = batch_data.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc = compute_acc(CM)
        mIoU = compute_mIoU(CM)

        loop.set_description(f'Train Epoch [{e+1}/{epoch}]')
        loop.set_postfix({"train_loss":loss_show.average.item(),
                          "train_accuracy": str(round(acc*100, 2)) + "%",
                          "train_mIoU": str(round(mIoU*100, 2)) + "%"})

    return acc, mIoU, loss_show.average.item()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# valid model
def valid_epoch(model, val_loader, criterion, e, epoch, device, num_classes):
    loss_show = AverageMeter()
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    loop = tqdm(enumerate(val_loader), total = len(val_loader))
    with torch.no_grad():
        for batch_idx, (batch_data, batch_label) in loop:
            batch_data = batch_data.to(device).float()
            batch_label = batch_label.to(device).long()

            batch_prediction = model(batch_data)
            loss = criterion(batch_prediction, batch_label)

            batch_prediction = F.softmax(batch_prediction, dim=1)
            batch_prediction = torch.argmax(batch_prediction, dim=1)
            # calculate the accuracy
            CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
            CM = CM + CM_batch
            n = batch_data.shape[0]

            # update the loss and the accuracy 
            loss_show.update(loss.data, n)
            acc = compute_acc(CM)
            mIoU = compute_mIoU(CM)

            loop.set_description(f'Val Epoch [{e+1}/{epoch}]')
            loop.set_postfix({"val_loss":loss_show.average.item(),
                            "val_accuracy": str(round(acc*100, 2)) + "%",
                            "val_mIoU": str(round(mIoU*100, 2)) + "%"})

    return CM, acc, mIoU, loss_show.average.item()
#-------------------------------------------------------------------------------