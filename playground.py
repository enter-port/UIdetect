import os
import cv2
import math
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

def show_bbox(image, coords, names):
    image, coords = image_resize(image, (1080, 1920), np.array(coords))
    for i in range(len(coords)):
        x_min, y_min, x_max, y_max = coords[i]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, names[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
    
    # Save the result
    cv2.imshow("image", edges)
    cv2.waitKey(0)


def main():   
    data_path = "./data"
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)
                coords, names = parse_xml(xml_path, CLASS_NAME=None)
                if 'level_1' in names:
                    img_path = xml_path.replace(".xml", ".png")
                    if os.path.exists(img_path):
                        detect_edges(img_path)

if __name__  == "__main__":
    main()