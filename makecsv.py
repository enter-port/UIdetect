# -*- coding=utf-8 -*-
import os
import torch
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import csv
import pandas as pd
import numpy as np
import cv2
from model.centernet import CenterNet  # 确保路径正确
from dataset.dataset import UIDataset
from torch.utils.data import DataLoader

def load_model(checkpoint_path, device, category_path):
    """加载预训练的模型，并尝试从checkpoint或category文件加载类别信息"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果checkpoint是CenterNet模型实例，则尝试找到类别数量
    if isinstance(checkpoint, CenterNet):
        cls_head = checkpoint.head.cls_head
        
        # 找到最后一层卷积层以确定类别数量
        num_cat = None
        for module in reversed(cls_head):
            if isinstance(module, nn.Conv2d):
                num_cat = module.out_channels
                break
        
        if num_cat is None:
            raise ValueError("Could not find Conv2d layer in cls_head to determine number of categories.")
    else:
        raise TypeError("The checkpoint file should be a CenterNet model instance.")

    # 加载类别信息（优先使用category文件）
    categories = []
    with open(category_path, 'r', encoding='utf-8') as file:
        for line in file:
            cat = line.strip()
            if cat:  # 忽略空行
                categories.append(cat)

    num_cat_from_file = len(categories)
    if num_cat != num_cat_from_file:
        print(f"Warning: Mismatch between number of categories in model ({num_cat}) and in file ({num_cat_from_file}). Using file categories.")
        num_cat = num_cat_from_file

    # 初始化模型并加载权重
    model = CenterNet(backbone="resnet101", num_classes=num_cat)
    model.load_state_dict(checkpoint.state_dict())
    model.to(device)
    model.eval()

    return model, categories

def convert_predictions_to_boxes(hms_pred, whs_pred, offsets_pred, input_shape, categories):
    """Convert model predictions to bounding boxes."""
    boxes = []
    threshold = 0  # Confidence threshold for detection
    
    # Ensure the shapes are correct
    try:
        num_classes, output_height, output_width = hms_pred.shape
        _, wh_output_height, wh_output_width = whs_pred.shape
        _, offset_output_height, offset_output_width = offsets_pred.shape
    except ValueError:
        print(f"Shapes: hms_pred={hms_pred.shape}, whs_pred={whs_pred.shape}, offsets_pred={offsets_pred.shape}")
        raise ValueError("Mismatch in expected dimensions for hms_pred, whs_pred, or offsets_pred.")
        
    if (output_height != wh_output_height or output_width != wh_output_width or 
        output_height != offset_output_height or output_width != offset_output_width):
        raise ValueError("Mismatch in output dimensions between hms_pred, whs_pred, and offsets_pred.")

    print(f"Heatmap shape: {hms_pred.shape}")
    print(f"Width/Height shape: {whs_pred.shape}")
    print(f"Offset shape: {offsets_pred.shape}")
    print(f"Number of classes: {num_classes}, Output height: {output_height}, Output width: {output_width}")

    for c in range(num_classes):  # Iterate over classes
        for h in range(output_height):  # Iterate over height
            for w in range(output_width):  # Iterate over width
                confidence = hms_pred[c, h, w]
                if confidence > threshold:
                    width, height = whs_pred[:, h, w]  # Access width and height
                    offset_x, offset_y = offsets_pred[:, h, w]  # Access offsets
                    
                    center_x = w + offset_x
                    center_y = h + offset_y
                    
                    x_min = center_x - width / 2
                    y_min = center_y - height / 2
                    x_max = center_x + width / 2
                    y_max = center_y + height / 2
                    
                    boxes.append((x_min, y_min, x_max, y_max, confidence, c))
    
    return boxes

def save_predictions_to_csv(predictions, output_csv, image_names, categories):
    """将预测结果保存为CSV文件，确保格式符合eval要求"""
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'predictions'])
        
        for img_name, preds in zip(image_names, predictions):
            prediction_details = []
            for pred in preds:
                x_min, y_min, x_max, y_max, confidence, class_idx = pred
                class_name = categories[class_idx] if class_idx < len(categories) else "unknown"
                bbox_info = f"{x_min} {y_min} {x_max - x_min} {y_max - y_min} {confidence} {class_name}"
                prediction_details.append(bbox_info)
            writer.writerow([img_name, ' '.join(prediction_details)])

def infer_and_save_results(model, dataloader, output_csv, device, input_shape, categories):
    predictions = []
    image_names = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            images, batch_hm, batch_wh, batch_offset, batch_offset_mask = batch
            images = images.float().to(device)
            
            # Get model predictions
            outputs = model(images, mode='inference')
            print("Model outputs shapes:")
            for i, output in enumerate(outputs):
                print(f"Output {i} shape: {output.shape}")

            # Ensure outputs are numpy arrays and remove batch dimension correctly
            hms_pred, whs_pred, offsets_pred = [output.cpu().numpy()[0] for output in outputs[:3]]

            boxes = convert_predictions_to_boxes(
                hms_pred,  # Already removed batch dimension, shape should be (10, 320, 480)
                whs_pred,  # Already removed batch dimension, shape should be (2, 320, 480)
                offsets_pred,  # Already removed batch dimension, shape should be (2, 320, 480)
                input_shape,
                categories
            )
                
            preds = [
                [x_min, y_min, x_max, y_max, score, class_idx]
                for x_min, y_min, x_max, y_max, score, class_idx in boxes
            ]
            predictions.append(preds)

            # 注意：这里需要确保你可以从dataloader或batch中获取图像名
            # 如果不能直接获取，可能需要在dataset中加入这个信息
            image_name = f"image_{len(image_names)}"  # 使用默认名称，如果无法获取真实名称
            image_names.append(image_name)
    
    save_predictions_to_csv(predictions, output_csv, image_names, categories)

if __name__ == "__main__":
    # 设置参数
    checkpoint_path = './logs/2024-12-17-18-14-2/weights/epoch=149_loss=1.3633_val_loss=11.1868.pt'  # 替换为你的checkpoint路径
    output_csv = 'predictions.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (1280, 1920)  # 根据你的模型输入尺寸调整
    category_path = "./data/categories.txt"  # 类别信息文件路径

    # 加载模型和类别信息
    model, categories = load_model(checkpoint_path, device, category_path)

    # 准备测试集 DataLoader
    test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 执行推理并将结果保存到CSV文件
    infer_and_save_results(model, test_loader, output_csv, device, input_shape, categories)