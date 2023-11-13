import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def output_metrics(prediction, label, num_classes):
    prediction = prediction.flatten()
    label = label.flatten()
    CM = confusion_matrix(label, prediction, labels=np.array(range(num_classes)))
    accuracy = accuracy_score(label, prediction)
    weighted_recall = recall_score(label, prediction, average="weighted", zero_division=0)
    weighted_precision = precision_score(label, prediction, average="weighted", zero_division=0)
    weighted_f1 = f1_score(label, prediction, average="weighted", zero_division=0)
    
    # compute mean iou
    np.seterr(divide="ignore", invalid="ignore")
    intersection = np.diag(CM)
    ground_truth_set = CM.sum(axis=1)
    predicted_set = CM.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    mIoU = np.mean(IoU)

    return CM, accuracy, weighted_recall, weighted_precision, weighted_f1, IoU, mIoU

def compute_mIoU(CM):
    np.seterr(divide="ignore", invalid="ignore")
    intersection = np.diag(CM)
    ground_truth_set = CM.sum(axis=1)
    predicted_set = CM.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    mIoU = np.mean(IoU)
    return mIoU

def compute_acc(CM):
    np.seterr(divide="ignore", invalid="ignore")
    TP = np.sum(np.diag(CM))
    Sum = np.sum(CM)
    acc = TP/Sum
    return acc

if __name__ == "__main__":
    pred = np.array([0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5])
    true = np.array([0,1,1,1,2,2,2,3,4,3,3,5,4,4,4,5,5])
    CM, accuracy, weighted_recall, weighted_precision, weighted_f1, IoU, MIoU = output_metrics(pred, true)
    print(accuracy, weighted_recall, weighted_precision, weighted_f1, MIoU)
    print(compute_acc(CM))
    print(compute_mIoU(CM))
    print(CM)
    print(IoU)