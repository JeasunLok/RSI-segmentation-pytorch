import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# def output_metrics(prediction, label, num_classes):
#     prediction = prediction.flatten()
#     label = label.flatten()
#     CM = confusion_matrix(label, prediction, labels=np.array(range(num_classes)))
#     accuracy = accuracy_score(label, prediction)
#     weighted_recall = recall_score(label, prediction, average="weighted", zero_division=0)
#     weighted_precision = precision_score(label, prediction, average="weighted", zero_division=0)
#     weighted_f1 = f1_score(label, prediction, average="weighted", zero_division=0)
    
#     # compute mean iou
#     np.seterr(divide="ignore", invalid="ignore")
#     intersection = np.diag(CM)
#     ground_truth_set = CM.sum(axis=1)
#     predicted_set = CM.sum(axis=0)
#     union = ground_truth_set + predicted_set - intersection
#     IoU = intersection / union.astype(np.float32)
#     mIoU = np.mean(IoU)

#     return CM, accuracy, weighted_recall, weighted_precision, weighted_f1, IoU, mIoU

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

def compute_metrics(CM):
    np.seterr(divide="ignore", invalid="ignore")
    num_classes = CM.shape[0]
    GT_array = np.sum(CM, axis=0)
    TP_array = np.diag(CM)
    Recall_array = np.array([])
    Precision_array = np.array([])
    F1_array = np.array([])

    for i in range(num_classes):
        TP = TP_array[i]
        FP = np.sum(CM[i,:])-TP
        FN = np.sum(CM[:,i])-TP
        TN = np.sum(CM)-TP-FP-FN
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        F1 = 2*Recall*Precision/(Recall+Precision)
        Recall_array = np.append(Recall_array, Recall)
        Precision_array = np.append(Precision_array, Precision)
        F1_array = np.append(F1_array, F1)

    weighted_Recall = np.sum(GT_array*Recall_array)/np.sum(CM)
    weighted_Precision = np.sum(GT_array*Precision_array)/np.sum(CM)
    weighted_F1 = np.sum(GT_array*F1_array)/np.sum(CM)

    return weighted_Recall, weighted_Precision, weighted_F1


if __name__ == "__main__":
    pred = np.array([0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,4])
    true = np.array([0,1,1,1,2,2,2,3,4,3,3,5,4,4,4,5,5])
    CM = confusion_matrix(pred, true)
    weighted_Recall, weighted_Precision, weighted_F1 = compute_metrics(CM)
    print(weighted_Recall, weighted_Precision, weighted_F1)