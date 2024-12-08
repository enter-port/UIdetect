import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2

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

data_path = "./data/traindata"
subject = os.listdir(data_path)
all_names = list()
for sub in subject:
    files = os.listdir(data_path + "/" + sub)
    for filename in files:
        file_root, file_extension = os.path.splitext(filename)
        if file_extension == ".xml":
            _, names = parse_xml(data_path + "/" + sub + "/"+ filename, "AJHFIUGEUJNIFUO")
            for name in names:
                if name == "level_2":
                    print(sub, filename)
                if name not in all_names:
                    all_names.append(name)
                    
print(all_names)
    
test_path = "./data/traindata/coredraw/frame_3_2"
xml_path = test_path + ".xml"
img_path = test_path + ".png"
coords = parse_xml(xml_path, "FILLING")

img = cv2.imread(img_path)
for rect in coords:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4)) 
cv2.imshow('img', img)
cv2.waitKey()
