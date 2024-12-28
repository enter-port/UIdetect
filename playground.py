import os
import cv2
import math
import json
import torch
import shutil
import numpy as np
import torch.optim as optim
import xml.etree.ElementTree as ET
from tqdm import tqdm
from datetime import datetime
from model.centernet import CenterNet
from dataset.dataset import UIDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.train_utils import get_summary_image
from utils.data_utils import image_resize
from utils.data_utils import preprocess_input, recover_input

def parse_xml(xml_path, CLASS_NAME):
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
        # if CLASS_NAME == name:
        coords.append([x_min, y_min, x_max, y_max])
        names.append(name)
    return coords, names

"""train_data_path = "./data/traindata/winmediaplayer"
files = os.listdir(train_data_path)
for filename in files:
    file_root, file_extension = os.path.splitext(filename)
    if file_extension == ".png":
        img = cv2.imread(train_data_path + "/" + filename)
        print(img.shape)"""

"""
coredraw: 3120 * 2080
jiguang: 1919 * 1079
onedrive: 2080 * 3120
skype: 3120 * 2080
winmediaplayer: 1919 * 1079
"""

def show_bbox(image, coords, pboxes, names=None, output_dir=None):
    image, coords, pboxes = image_resize(image, (1080, 1920), np.array(coords), np.array(pboxes))   
    for i in range(len(coords)):
        x_min, y_min, x_max, y_max = coords[i]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # cv2.putText(image, names[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for i in range(len(pboxes)):
        x_min, y_min, x_max, y_max = pboxes[i]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
def detect_edges(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image is loaded successfully
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)
    
    cv2.imshow("image", edges)
    cv2.waitKey(0)   

def main():   
    """data_path = "./data"
    valid_names = ["level_0", "level_1", "level_2"]
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)
                coords, names = parse_xml(xml_path, CLASS_NAME=None)
                filtered_coords = [coord for coord, name in zip(coords, names) if name in valid_names]
                filtered_names = [name for name in names if name in valid_names]
                if filtered_coords:
                    img_path = xml_path.replace(".xml", ".png")
                    img = cv2.imread(img_path)
                    subdir = os.path.relpath(root, data_path)
                    output_dir = os.path.join("./data_level", subdir)
                    os.makedirs(output_dir, exist_ok=True)
                    output_img_path = os.path.join(output_dir, os.path.basename(img_path))
                    show_bbox(img, filtered_coords, filtered_names, output_img_path)"""
                        
    path = "./data/coredraw/frame_3_2"
    
    image = cv2.imread(path + ".png")
    coords, names = parse_xml(path + ".xml", CLASS_NAME=None)
    with open(path + ".json", 'r') as f:
        pre_bboxes = json.load(f)
        
    image, coords, pre_bboxes = image_resize(image, (1080, 1920), np.array(coords), np.array(pre_bboxes))   

    corners = []
    for box in pre_bboxes:
        x0, y0, x1, y1 = box
        corners.extend([(x0, y0), (x0, y1), (x1, y0), (x1, y1)])

    non_overlapping_boxes = []
    overlapping_boxes = []

    for box in pre_bboxes:
        x0, y0, x1, y1 = box
        is_overlapping = False
        for (cx, cy) in corners:
            if x0 < cx < x1 and y0 < cy < y1:
                is_overlapping = True
                break
        if is_overlapping:
            overlapping_boxes.append(box)
        else:
            non_overlapping_boxes.append(box)
            

    for box in non_overlapping_boxes:
        x0, y0, x1, y1 = box
        center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
        new_width, new_height = (x1 - x0) * 2, (y1 - y0) * 2
        new_x0, new_y0 = center_x - new_width // 2, center_y - new_height // 2
        new_x1, new_y1 = center_x + new_width // 2, center_y + new_height // 2

        new_x0, new_y0 = max(new_x0, 0), max(new_y0, 0)
        new_x1, new_y1 = min(new_x1, image.shape[1]), min(new_y1, image.shape[0])

        new_rect = image[new_y0:new_y1, new_x0:new_x1]
        avg_color = new_rect.mean(axis=(0, 1))

        image[y0:y1, x0:x1] = avg_color
        
    for i in range(len(overlapping_boxes)):
        x_min, y_min, x_max, y_max = overlapping_boxes[i]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # cv2.putText(image, names[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for i in range(len(non_overlapping_boxes)):
        x_min, y_min, x_max, y_max = non_overlapping_boxes[i]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  
    for i in range(len(coords)):
        x_min, y_min, x_max, y_max = coords[i]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
          
    cv2.imshow("image", image)
    cv2.waitKey(0)



if __name__  == "__main__":
    main()