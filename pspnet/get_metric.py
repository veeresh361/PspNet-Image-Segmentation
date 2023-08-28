import os
import cv2
import numpy as np
import torch
DEVICE='cuda'

def calculate_iou(pred_mask, true_mask):
    # print(pred_mask.shape)
    # print(true_mask.shape)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    iou = intersection / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    return iou

def predict_image(model, image):
    model.eval()
    with torch.no_grad():
        logits_mask = model(image.to(DEVICE))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5).float()
    return pred_mask.squeeze(0).cpu().numpy()

def calculate_precision_recall(pred_mask, true_mask):
    TP = np.logical_and(pred_mask, true_mask).sum()
    FP = np.logical_and(pred_mask, 1 - true_mask).sum()
    FN = np.logical_and(1 - pred_mask, true_mask).sum()

    precision = TP / (TP + FP + 1e-6)  # Adding a small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-6)

    return precision, recall

def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Handle the case of an empty list
    
    total = sum(numbers)
    average = total / len(numbers)
    return average


actual_mask_path="/home/veeresh/segmentation/pspnet/actual_masks/"
predicted_mask_path="/home/veeresh/segmentation/pspnet/predicted_mask"
iocList=[]
pList=[]
rList=[]
for i in os.listdir(actual_mask_path):
    
    actualMask=cv2.imread(actual_mask_path+'/'+i)
    predictedMask=cv2.imread(predicted_mask_path+'/'+i)
    res=calculate_iou(predictedMask,actualMask)
    p,r=calculate_precision_recall(predictedMask,actualMask)
    iocList.append(res)
    pList.append(p)
    rList.append(r)

print(calculate_average(iocList),calculate_average(pList),calculate_average(rList))

