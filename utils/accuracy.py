import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)
	# 在输出结果中取前maxk个最大概率作为预测结果，并获取其下标，当topk=(1, 5)时取5就可以了。
    _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
    
    # 将得到的k个预测结果的矩阵进行转置，方便后续和label作比较
    pred = pred.T
    # 将label先拓展成为和pred相同的形状，和pred进行对比，输出结果
    correct = torch.eq(pred, label.contiguous().view(1,-1).expand_as(pred))
    res = []

    for k in topk:
    	# 取前k个预测正确的结果进行求和
        correct_k = correct[:k].contiguous().view(-1).float().sum(dim=0, keepdim=True)
        # 计算平均精度， 将结果加入res中
        res.append(correct_k*100/batch_size)
    return res

def output_metrics(prediction, label):
    CM = confusion_matrix(label, prediction)
    weighted_recall = recall_score(label, prediction, average="weighted")
    weighted_precision = precision_score(label, prediction, average="weighted")
    weighted_f1 = f1_score(label, prediction, average="weighted")
    
    # ytrue, ypred is a flatten vector
    y_pred = prediction.flatten()
    y_true = label.flatten()
    current = confusion_matrix(y_true, y_pred)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    MIoU = np.mean(IoU)

    return CM, weighted_recall, weighted_precision, weighted_f1, IoU, MIoU

if __name__ == "__main__":
    pred = np.array([0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5])
    true = np.array([0,1,1,1,2,2,2,3,4,3,3,5,4,4,4,5,5])
    CM, weighted_recall, weighted_precision, weighted_f1, IoU, MIoU = output_metrics(pred, true)
    print(weighted_recall, weighted_precision, weighted_f1, MIoU)
    print(CM)
    print(IoU)