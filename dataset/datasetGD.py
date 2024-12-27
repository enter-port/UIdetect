'''
A modified version of the original dataset
Modified for our new pipeline
'''
import torch
import os
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.data_utils import parse_xml, image_resize, preprocess_input

class UIDataset(Dataset):
    def __init__(self, data_path, category_path, input_shape=(1080, 1080)):
        super(UIDataset, self).__init__()
        self.input_shape = input_shape

        self.category = []
        with open(category_path, 'r', encoding='utf-8') as file:
            for line in file:
                cat = line.strip()
                self.category.append(cat)
        self.num_cat = len(self.category)

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

        self.data_names = datanames

        self.images = []
        pbar = tqdm(self.data_names)
        '''
        change image into following format:
        (C,H,W)
        image shape in input_shape(note: no stride used here! Simply add stride if want)
        pixel value have been divided by 255
        '''
        for name in pbar:
            img_path = data_path + name + ".png"
            image = Image.open(img_path).convert("RGB")
            image = image.resize(self.input_shape) 
            image_array = np.array(image) / 255.0 
            image_tensor = torch.tensor(image_array).permute(2, 0, 1)  
            
            self.images.append(image_tensor)

        self.length = len(self.images)
        print(f"Dataset Initialize Done. {self.length} objects in total.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.images[idx]