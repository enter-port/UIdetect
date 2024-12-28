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
from utils.data_utils import parse_xml, gaussian_radius, draw_gaussian, image_resize, preprocess_input, data_augmentation, cal_feature

class UIDataset(Dataset):
    def __init__(self, data_path, input_shape=(1080, 1080), is_train=True, split_radio=0.8, target="class"):
        super(UIDataset, self).__init__()
        self.stride = 4
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.is_train = is_train
        
        self.target = target
        self.category =["clickable", "selectable", "scrollable", "disabled"] if target == "class" else ["level_0", "level_1", "level_2"]
        self.num_cat = len(self.category)
        
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
                    if "/{}/{}".format(sub, file_root) not in datanames:
                        datanames.append("/{}/{}".format(sub, file_root))
            random.shuffle(datanames)
            train_datanames = datanames[:int(len(datanames) * split_radio)]
            test_datanames = datanames[int(len(datanames) * split_radio):]
            data_dict = {"train": train_datanames, "test": test_datanames}
            with open(data_path + "/train_test_split.json", 'w') as f:
                json.dump(data_dict, f, indent=4)
            print("Split stored at {}.".format(data_path + "/train_test_split.json"))
        
        else:
            print("Split exists. Using split at {}".format(data_path + "/train_test_split.json"))
            with open(data_path + "/train_test_split.json", 'r') as f:
                data = json.load(f)
            train_datanames = data["train"]
            test_datanames = data["test"]
            
        data_names = train_datanames if is_train else test_datanames
        
        self.images = []
        self.bboxes = []
        self.file_names = [] 
        if self.target == "class":
            self.pboxes = []
        
        pbar = tqdm(data_names)
        for name in pbar:
            img_path = data_path + name + ".png"
            xml_path = data_path + name + ".xml"
            pre_path = data_path + name + ".json"
        
            image = Image.open(img_path)
            image = np.array(image)[:, :, :3]
            coords, names = parse_xml(xml_path)

            root_coords = None
            new_coords = []
            new_names = []
            
            for coord, name in zip(coords, names):
                if name.lower() == "root":
                    root_coords = coord
                elif name.lower() in self.category:
                    new_coords.append(coord)
                    new_names.append(name.lower())
            
            if root_coords is not None:
                x0, y0, x1, y1 = root_coords
                image = image[y0:y1, x0:x1]
                
                for i in range(len(new_coords)):
                    new_coords[i] = [
                        new_coords[i][0] - x0,
                        new_coords[i][1] - y0,
                        new_coords[i][2] - x0,
                        new_coords[i][3] - y0
                    ]
                
                if self.target == "class":
                    new_pboxes = []
                    with open(pre_path, 'r') as f:
                        pre_bboxes = json.load(f)
                    for i in range(len(pre_bboxes)):
                        box = [
                            pre_bboxes[i][0] - x0,
                            pre_bboxes[i][1] - y0,
                            pre_bboxes[i][2] - x0,
                            pre_bboxes[i][3] - y0
                        ]
                        if all(coord > 0 for coord in box):
                            new_pboxes.append(box)
                        
            coords = new_coords
            names = new_names
            pre_bboxes = new_pboxes
            
            if names == []:
                continue
                       
            if self.target == "class":
                image, bbox, pbbox = image_resize(image, input_shape, np.array(coords), np.array(pre_bboxes))
                image = preprocess_input(image)
                
                category_indices = np.array([int(self.category.index(name.lower())) for name in names])
                bbox = np.column_stack((bbox, category_indices))
                
                pre_category_indices = np.array([0 for _ in range(len(pbbox))])
                pbbox = np.column_stack((pbbox, pre_category_indices))
                
                self.images.append(image)
                self.bboxes.append(bbox)
                self.pboxes.append(pbbox)
            
            else:
                image, bbox = image_resize(image, input_shape, np.array(coords))
                image = preprocess_input(image)
                assert len(coords) == len(names)
                
                category_indices = np.array([int(self.category.index(name.lower())) for name in names])
                bbox = np.column_stack((bbox, category_indices))
                
                self.images.append(image)
                self.bboxes.append(bbox)
                
            self.file_names.append(name + ".png") 
            
        self.length = len(self.images)
        assert len(self.images) == len(self.bboxes)
        print("{} Dataset for targrt {} Initialize Done. {} objects in total.".format("Train" if is_train else "Test", target, self.length))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):        
        image = self.images[idx]
        bbox = np.array(self.bboxes[idx])
        file_name = self.file_names[idx]
        
        batch_hm, batch_wh, batch_offset, batch_offset_mask = cal_feature(image, bbox, self.output_shape, self.num_cat, self.stride)
        
        if self.target == "class":
            pbox = self.pboxes[idx]
            pre_hm, pre_wh, pre_offset, _ = cal_feature(image, pbox, self.output_shape, 1, self.stride) 
            return image, batch_hm, batch_wh, batch_offset, batch_offset_mask, pre_hm, pre_wh, pre_offset, file_name
        else:
            return image, batch_hm, batch_wh, batch_offset, batch_offset_mask, file_name
    

        