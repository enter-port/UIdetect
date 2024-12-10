import os 
import math
import json
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset
from utils.data_utils import parse_xml, gaussian_radius, draw_gaussian, image_resize, preprocess_input

class UIDataset(Dataset):
    def __init__(self, data_path, category_path, input_shape=(1080, 1080), is_train=True, split_radio=0.8):
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
        
        # if no pre-defined train-test split exists, we create a random one
        if not os.path.exists(data_path + "/train_test_split.json"):
            print("No train-test split. We create one at random.")
            datanames = []
            subject = os.listdir(data_path)
            for sub in subject:
                if not os.path.isdir(data_path + "/" + sub):
                    continue
                files = os.listdir(data_path + "/" + sub)
                for filename in files:
                    file_root, file_extension = os.path.splitext(filename)
                    print("/{}/{}".format(sub, file_root))
                    if "/{}/{}".format(sub, file_root) not in datanames:
                        datanames.append("/{}/{}".format(sub, file_root))
            random.shuffle(datanames)
            train_datanames = datanames[:int(len(datanames) * split_radio)]
            test_datanames = datanames[int(len(datanames) * split_radio):]
            data_dict = {"train": train_datanames, "test": test_datanames}
            with open(data_path + "/train_test_split.json", 'w') as f:
                json.dump(data_dict, f, indent=4)
            print("Split strored at {}.".format(data_path + "/train_test_split.json"))
        
        # else we directly read for train_test_split.json
        else:
            print("Split exists. Using split at{}".format(data_path + "/train_test_split.json"))
            with open(data_path + "/train_test_split.json", 'r') as f:
                data = json.load(f)
            train_datanames = data["train"]
            test_datanames = data["test"]
            
        # Set the files we want to read according to IS_TRAIN
        data_names = train_datanames if is_train else test_datanames
            
        # read in images, boundboxes and names; name is expressed as index of category
        self.images = []
        self.bboxes = []    # the bboxes should be a list of bbox, bbox:[[x0, y0, x1, y1, name], [...]]
        pbar = tqdm(data_names)
        for name in pbar:
            img_path = data_path + name+ ".png"
            xml_path = data_path + name + ".xml"
        
            # read in .png image and convert it into desired size
            image = Image.open(img_path)
            image = np.array(image)[:, :, :3]
            
            # read in bbox and name from .xml file
            coords, names = parse_xml(xml_path)
            image, bbox = image_resize(image, input_shape, np.array(coords))
            image = preprocess_input(image)
            assert len(coords) == len(names)
            
            # add category index to bbox
            category_indices = np.array([int(self.category.index(name.lower())) for name in names])
            bbox = np.column_stack((bbox, category_indices))
            
            self.images.append(image)
            self.bboxes.append(bbox)
            
        self.length = len(self.images)
        assert len(self.images) == len(self.bboxes)
        assert len(self.images) == len(data_names)
        print("{} Dataset Initialize Done. {} objects in total.".format("Train" if is_train else "Test", self.length))
        
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
        