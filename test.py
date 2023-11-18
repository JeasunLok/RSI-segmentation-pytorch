import numpy as np
import torch
from tqdm import tqdm
from utils.accuracy import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------------------------------
# test model
def test_epoch(model, test_loader, device, num_classes):
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    loop = tqdm(enumerate(test_loader), total = len(test_loader))
    with torch.no_grad():
        for batch_idx, (batch_data, batch_label) in loop:
            batch_data = batch_data.to(device).float()
            batch_label = batch_label.to(device).long()

            batch_prediction = model(batch_data)

            batch_prediction = F.softmax(batch_prediction, dim=1)
            batch_prediction = torch.argmax(batch_prediction, dim=1)

            # calculate the accuracy
            CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
            CM = CM + CM_batch

            # update the loss and the accuracy 
            acc = compute_acc(CM)
            mIoU = compute_mIoU(CM)

            loop.set_description(f'Test Epoch')
            loop.set_postfix({"test_accuracy": str(round(acc*100, 2)) + "%",
                              "test_mIoU": str(round(mIoU*100, 2)) + "%"})

    return CM, acc, mIoU