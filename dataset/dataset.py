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
        
        if self.is_train:  
            image, bbox = self.data_augmentation(image, bbox)
        
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

    def data_augmentation(self, image, bboxes):
        if random.random() < 0.5:
            image, bboxes = self.random_horizontal_flip(image, bboxes)
        # if random.random() < 0.5:
        #     image, bboxes = self.random_vertical_flip(image, bboxes)
        if random.random() < 0.5:
            image, bboxes = self.random_crop(image, bboxes)
        if random.random() < 0.5:
            image, bboxes = self.random_translate(image, bboxes)

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

    def random_horizontal_flip(self, image, bboxes):
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        image = np.array(image)

        if bboxes.size != 0:
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_vertical_flip(self, image, bboxes):
        h, _, _ = image.shape
        image = image[::-1, :, :]
        image = np.array(image)

        if bboxes.size != 0:
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]

        return image, bboxes

    def random_crop(self, image, bboxes):
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

    def random_translate(self, image, bboxes):
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
        image = cv.warpAffine(image, M, (w, h), borderValue=(128, 128, 128))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes
       