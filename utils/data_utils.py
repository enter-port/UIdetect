# -*- coding=utf-8 -*-
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import cv2
import random
import math
from PIL import Image, ImageDraw, ImageEnhance

def parse_csv(csv_path, CLASS_NAME):
    '''
    输入：
        csv_path: csv的文件路径
    输出：
        从csv文件中提取信息, 格式为[[img_name, [[x_min, y_min, x_max, y_max, confidence],...],
                                [img_name, [[x_min, y_min, x_max, y_max, confidence],...],...]
    '''
    results = list()
    coord_confidence_s = list()
    img_name_2 = str()
    with open(csv_path) as csvfile:
        mLines = csvfile.readlines()
        for mStr in mLines:
            row = mStr.split(",")
            img_name_1 = row[0]
            coord = row[1].split()
            x_min = int(coord[0])
            y_min = int(coord[1])
            x_max = x_min + int(coord[2])
            y_max = y_min + int(coord[3])
            confidence = float(row[2])  # 取结果文档中的分数2-head 4-smoke 6-phone
            name = str(row[3].split()[0])  # 取结果文档中的标签
            if img_name_1 != img_name_2 and mStr != mLines[0] and coord_confidence_s:
                result = [img_name_2, coord_confidence_s]
                results.append(result)
                coord_confidence_s = []
            img_name_2 = img_name_1
            coord_confidence = [x_min, y_min, x_max, y_max, confidence]
            # person，vehicle，rider，tricycles
            if CLASS_NAME == name:
                coord_confidence_s.append(coord_confidence)
            if mStr == mLines[-1] and coord_confidence_s:
                result = [img_name_2, coord_confidence_s]
                results.append(result)
    return results


def parse_xml(xml_path, CLASS_NAME=None):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, coords格式为[[x_min, y_min, x_max, y_max],...]
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    names = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(float(box.find('xmin').text))
        y_min = int(float(box.find('ymin').text))
        x_max = int(float(box.find('xmax').text))
        y_max = int(float(box.find('ymax').text))
        width = x_max - x_min
        height = y_max - y_min
        names.append(name)
        coords.append([x_min, y_min, x_max, y_max])
    return coords, names

def readCsv(filename, usecols, header=None):
    try:
        data_csv=pd.read_csv(filename,sep=None,engine='python',usecols=usecols,header=header, encoding='utf-8')
    except:
        data_csv=pd.read_csv(filename,sep=None,engine='python',usecols=usecols,header=header)
    return data_csv


def fileToArray(pd_data):
    data = pd_data.dropna(axis=0, how='any')
    data_new = data.values
    data_array = np.array(data_new)
    return data_array


def err_drawing(CLASS_NAME, ERR_S, img_paths, outimg_paths):
    res = readCsv('error_analysis_{}_{}.csv'.format(CLASS_NAME, ERR_S), (0, 1, 2, 3), header=0)
    res = fileToArray(res)
    for cells in res:
        # 0:图片名字 1:坐标:x y w h 2:分数 3:类名
        jpg_name = cells[0]
        threshold = cells[2]
        class_name = cells[3]
        # 获取坐标
        coordinate = cells[1]
        coordinate = (coordinate.split(' '))
        x, y, w, h = int(coordinate[0]), int(coordinate[1]), (int(coordinate[2]) - int(coordinate[0])), (
                    int(coordinate[3]) - int(coordinate[1]))
        # 获取对应图片路径，并按照坐标画框
        img_path = os.path.join(img_paths , jpg_name)
        save_jpg_name = outimg_paths + jpg_name
        if os.path.exists(save_jpg_name):
            img = cv2.imread(save_jpg_name)
        else:
            img = cv2.imread(img_path)
        # 颜色坐标:绿色0,255,0 红色0,0,255  蓝色255,0,0 黄色255,255,0
        colour = (0,0,255)
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), colour, 2)
        # biaoqian = class_name + str(threshold)
        cv2.putText(img, str(threshold), (x, (y + int(h / 2))), cv2.FONT_HERSHEY_COMPLEX, 1, colour, 2)
        # 存放路径
        save_jpg_name = os.path.join(outimg_paths ,jpg_name)
        cv2.imwrite(save_jpg_name, img)
        
def gaussian_radius(det_size, min_overlap=0.7):
    """
    Get gaussian circle radius.
    Args:
        det_size: (height, width)
        min_overlap: overlap minimum

    Returns: radius

    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    """
    2D Gaussian function
    Args:
        shape: (diameter, diameter)
        sigma: variance

    Returns: h

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

def draw_gaussian(heatmap, center, radius, k=1):
    """
    Get a heatmap of one class
    Args:
        heatmap: The heatmap of one class(storage in single channel)
        center: The location of object center
        radius: 2D Gaussian circle radius
        k: The magnification of the Gaussian

    Returns: heatmap

    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def image_resize(image, target_size, gt_boxes=None, pre_gt_boxes=None):
    ih, iw = target_size

    h, w = image.shape[:2]

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded = cv2.copyMakeBorder(image_resized, dh, dh if (ih-nh)%2==0 else dh+1, 
                                      dw, dw if (iw-nw)%2==0 else dw+1, 
                                      cv2.BORDER_CONSTANT, value=(128, 128, 128))
    assert image_padded.shape[:2] == target_size
    if gt_boxes is None and pre_gt_boxes is None:
        return image_padded
    elif pre_gt_boxes is None:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes
    else:   
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        pre_gt_boxes[:, [0, 2]] = pre_gt_boxes[:, [0, 2]] * scale + dw
        pre_gt_boxes[:, [1, 3]] = pre_gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes, pre_gt_boxes
    
def preprocess_input(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    image = (image / 255. - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return image


def recover_input(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    image = np.transpose(image, (1, 2, 0))
    image = (image * std + mean) * 255

    return image.astype(np.uint8)

def data_augmentation(image, bboxes):
    if random.random() < 0.5:
        image, bboxes = random_horizontal_flip(image, bboxes)
    # if random.random() < 0.5:
    #     image, bboxes = random_vertical_flip(image, bboxes)
    if random.random() < 0.5:
        image, bboxes = random_crop(image, bboxes)
    if random.random() < 0.5:
        image, bboxes = random_translate(image, bboxes)
    if random.random() < 0.5:
        image = Image.fromarray(image)
        enh_bri = ImageEnhance.Brightness(image)
        # brightness = [1, 0.5, 1.4]
        image = enh_bri.enhance(random.uniform(0.6, 1.4))
        image = np.array(image)
    if random.random() < 0.5:
        image = Image.fromarray(image)
        enh_col = ImageEnhance.Color(image)
        # color = [0.7, 1.3, 1]
        image = enh_col.enhance(random.uniform(0.7, 1.3))
        image = np.array(image)
    if random.random() < 0.5:
        image = Image.fromarray(image)
        enh_con = ImageEnhance.Contrast(image)
        # contrast = [0.7, 1, 1.3]
        image = enh_con.enhance(random.uniform(0.7, 1.3))
        image = np.array(image)
    if random.random() < 0.5:
        image = Image.fromarray(image)
        enh_sha = ImageEnhance.Sharpness(image)
        # sharpness = [-0.5, 0, 1.0]
        image = enh_sha.enhance(random.uniform(0, 2.0))
        image = np.array(image)
    return image, bboxes

def random_horizontal_flip(image, bboxes):
    _, w, _ = image.shape
    image = image[:, ::-1, :]
    image = np.array(image)
    if bboxes.size != 0:
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
    return image, bboxes

def random_vertical_flip(image, bboxes):
    h, _, _ = image.shape
    image = image[::-1, :, :]
    image = np.array(image)
    if bboxes.size != 0:
        bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
    return image, bboxes

def random_crop(image, bboxes):
    if bboxes.size == 0:
        return image, bboxes
    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]
    crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
    crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
    crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
    crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
    image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return image, bboxes

def random_translate(image, bboxes):
    if bboxes.size == 0:
        return image, bboxes
    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]
    tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
    ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
    M = np.array([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h), borderValue=(128, 128, 128))
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return image, bboxes

def cal_feature(image, bbox_with_label, output_shape, num_cat, stride):
    batch_hm = np.zeros((output_shape[0], output_shape[1], num_cat), dtype=np.float32) 
    batch_wh = np.zeros((output_shape[0], output_shape[1], 2), dtype=np.float32)
    batch_offset = np.zeros((output_shape[0], output_shape[1], 2), dtype=np.float32)
    batch_offset_mask = np.zeros((output_shape[0], output_shape[1]), dtype=np.float32)
        
    labels = np.array(bbox_with_label[:, -1])
    bbox = np.array(bbox_with_label[:, :-1])
    
    if len(bbox) != 0:
        labels = np.array(labels, dtype=np.float32)
        bbox = np.array(bbox[:, :4], dtype=np.float32)
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]] / stride, a_min=0, a_max=output_shape[1])
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]] / stride, a_min=0, a_max=output_shape[0])
    
    for i in range(len(labels)):
        x1, y1, x2, y2 = bbox[i]
        cls_id = int(labels[i])
        
        h, w = y2 - y1, x2 - x1
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            
            ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            
            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
            
            batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h

            batch_offset[ct_int[1], ct_int[0]] = ct - ct_int

            batch_offset_mask[ct_int[1], ct_int[0]] = 1
            
    return batch_hm, batch_wh, batch_offset, batch_offset_mask