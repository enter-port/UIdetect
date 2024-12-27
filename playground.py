import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2
import PIL
from utils.data_utils import image_resize
from dataset.dino_dataset import UIDataset
from torch.utils.data import DataLoader
from utils.train_utils import *
from tqdm import tqdm

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
    image = PIL.Image.open("./data/coredraw/frame_1.png")    
    test_path = "./data/coredraw/frame_1"
    xml_path = test_path + ".xml"
    img_path = test_path + ".png"
    coords, names = parse_xml(xml_path, "FILLING")

    category_path = "./data/categories.txt"
    category = []
    with open(category_path, 'r', encoding='utf-8') as file:
        for line in file:
            cat = line.strip()
            category.append(cat)

    """image = np.array(image)
    image = image[:, :, :3]
    coords = np.array(coords)
    category_indices = np.array([int(category.index(name.lower())) for name in names])

    image, bbox = image_resize(image, (960, 1280), coords)
    bbox = np.column_stack((bbox, category_indices))       
    draw_image = draw_bbox(image, bbox, range(len(category)), category, show_name=True)

    print(draw_image.shape)"""
    # cv2.imshow('img', draw_image)
    # cv2.waitKey()
    
    input_shape = (1080, 1920)

    test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape, is_train=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=2, pin_memory=True)

    device = torch.device("cpu")
    
    tbar = tqdm(test_loader)
    
    for samples, targets in tbar:
        # print(targets)    
        samples = samples.to(device)
        # targets is a list a dict: (batch_size * len(dict))
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        batch_size = 1
        targets = [{k: v[i].to(device) for k, v in targets.items()} for i in range(batch_size)]
    # for images, hms_true, whs_true, offsets_true, offset_masks_true in test_loader:
    #     outputs_true = postprocess_output(hms_true, whs_true, offsets_true, 0.999, device)
    #     outputs_true = decode_bbox(outputs_true, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    #     images = images.cpu().numpy()
    #     for i in range(len(images)):
    #         image = images[i]

    #         output_true = outputs_true[i]
    #         if len(output_true) != 0:
    #             output_true = output_true.data.cpu().numpy()
    #             labels_true = output_true[:, 5]
    #             bboxes_true = output_true[:, :4]
    #         else:
    #             labels_true = []
    #             bboxes_true = []
            
    #         # image_true = draw_bbox(image, bboxes_true, labels_true, category, show_name=True)
    #         # cv2.imshow("img", image_true)
    #         # cv2.waitKey(0)
    #     break

if __name__  == "__main__":
    main()