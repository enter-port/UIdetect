import os 
import torch
import math
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from utils import parse_xml, gaussian_radius, draw_gaussian
from torch.utils.data.dataset import Dataset

class UIDataset(Dataset):
    def __init__(self, data_path, category_path, input_shape=(640, 480), is_train=True):
        super(UIDataset, self).__init__()
        self.stride = 4
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.is_train = is_train
        
        # read from .txt the categories we are training for
        self.category = []
        with open(category_path, 'r', encoding='utf-8') as file:
            for line in file:
                cat = line.strip()
                self.category.append(cat)
        self.num_cat = len(self.category)
        
        # read in images, boundboxes and names; name is expressed as index of category
        self.images = []
        self.bboxes = []    # the bboxes should be a list of bbox, bbox:[[x0, y0, x1, y1, name], [...]]
        transform = transforms.Compose([
            transforms.Resize((input_shape[0], input_shape[1])),
        ])
        subject = os.listdir(data_path)
        for sub in subject:
            if not os.path.isdir(data_path + "/" + sub):
                continue
            print("Initializing {}".format(sub))
            files = os.listdir(data_path + "/" + sub)
            for filename in files:
                file_root, file_extension = os.path.splitext(filename)
                # .png and .xml files have the same name, so in order to avoid repetition
                # we overlook the name of the .xml files and process the name of .png file twice
                if file_extension == ".xml":
                    continue 
                img_path = data_path + "/" + sub + "/" + file_root + ".png"
                xml_path = data_path + "/" + sub + "/" + file_root + ".xml"
                
                # read in .png image and convert it into desired size
                image = Image.open(img_path)
                original_width, original_height = image.size
                image = transform(image)
                image = np.array(image)[:, :, :-1]
                self.images.append(image)
                
                # read in bbox and name from .xml file
                coords, names = parse_xml(xml_path)
                bbox = []
                assert len(coords) == len(names)
                for i in range(len(coords)):
                    bbox.append([
                        int(coords[i][0] / original_width * input_shape[0]),
                        int(coords[i][1] / original_height * input_shape[1]),
                        int(coords[i][2] / original_width * input_shape[0]),
                        int(coords[i][3] / original_height * input_shape[1]),
                        int(self.category.index(names[i].lower()))
                    ])
                self.bboxes.append(bbox)
        self.length = len(self.images)
        assert len(self.images) == len(self.bboxes)
        print("Initialize Done. {} objects in total.".format(self.length))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # heat map
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_cat), dtype=np.float32) 
        # width and height  
        batch_wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        # offset
        batch_offset = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_offset_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        
        image = self.images[idx]
        bbox = np.array(self.bboxes[idx])
        
        labels = np.array(bbox[:, -1])
        bbox = np.array(bbox[:, :-1])
        
        if len(bbox) != 0:
            labels = np.array(labels, dtype=np.float32)
            bbox = np.array(bbox[:, :4], dtype=np.float32)
            bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]] / self.stride, a_min=0, a_max=self.output_shape[1])
            bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]] / self.stride, a_min=0, a_max=self.output_shape[0])
        
        for i in range(len(labels)):
            x1, y1, x2, y2 = bbox[i]
            cls_id = int(labels[i])
            
            h, w = y2 - y1, x2 - x1
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                
                # Calculate feature points
                ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                # Get gaussian heat map
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                
                # Assign ground truth height and width
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h

                # Assign center point offset
                batch_offset[ct_int[1], ct_int[0]] = ct - ct_int

                # Set the corresponding mask to 1
                batch_offset_mask[ct_int[1], ct_int[0]] = 1

        return image, batch_hm, batch_wh, batch_offset, batch_offset_mask
        
dev_set = UIDataset(data_path="./data/traindata", category_path="./data/traindata/categories.txt")
image, batch_hm, batch_wh, batch_offset, batch_offset_mask = dev_set.__getitem__(0)
print(image.shape)
print(batch_hm.shape)
print(batch_wh.shape)
print(batch_offset.shape)
print(batch_offset_mask.shape)

test_path = "./data/traindata/coredraw/frame_3_2"
xml_path = test_path + ".xml"
img_path = test_path + ".png"
coords, names = parse_xml(xml_path, "FILLING")
print(coords)
print(names)
assert len(coords) == len(names)

image = Image.open(img_path)
print(type(image))
original_width, original_height = image.size
target_width, target_height = 640, 480

transform = transforms.Compose([
    transforms.Resize((target_height, target_width)),
    transforms.ToTensor()
])

image = transform(image)
print(image.shape)
"""rect = [
    int(rect[0] / original_width * target_width),
    int(rect[1] / original_height * target_height),
    int(rect[2] / original_width * target_width),
    int(rect[3] / original_height * target_height)
]
print(rect)
draw = ImageDraw.Draw(image)
draw.rectangle(rect, outline="red", width=3)

image.show()"""