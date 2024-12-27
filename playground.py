import os
import math
import torch
import shutil
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from model.centernet import CenterNet
from dataset.dataset import UIDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.train_utils import get_summary_image

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


def main():   
    # run on gpu if cuda exists else cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on cuda")
    else:
        device = torch.device("cpu")
        print("Running on cpu")
        
    # hyper params
    global step
    step = 0
    input_shape = (1280, 1920)  # Please ensure the number you've put here can be devided by 32
    batch_size = 1
    init_lr = 1e-3
    end_lr = 1e-6
    freeze_epoch = 50
    unfreeze_epoch = 100
    
    # check data for number of classes(categories)
    # You can change category path here
    category_path = "./data/categories.txt"
    category = []
    with open(category_path, 'r', encoding='utf-8') as file:
        for line in file:
            cat = line.strip()
            category.append(cat)
    num_cat = len(category)
    
    # get CenterNet model
    model = CenterNet(backbone="resnet101", num_classes=num_cat)
    model.to(device)
    print("Model create successful.")
    
    # get train test dataset
    train_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape, is_train=True)
    image, batch_hm, batch_wh, batch_offset, batch_offset_mask, file_name = train_dataset.__getitem__(0)

if __name__  == "__main__":
    main()