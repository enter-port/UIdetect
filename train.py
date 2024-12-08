import os
import torch
from datetime import datetime
from model.centernet import CenterNet

def remove_dir_and_create_dir(dir_name, is_remove=True):
    """
    Make new folder, if this folder exist, we will remove it and create a new folder.
    Args:
        dir_name: path of folder
        is_remove: if true, it will remove old folder and create new folder

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "create.")
    else:
        if is_remove:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            print(dir_name, "create.")
        else:
            print(dir_name, "is exist.")

def main():
    # First create ./log dict to store the results
    # By default we will name the folder according to the current time
    now =  datetime.now()
    remove_dir_and_create_dir("./logs/{}-{}-{}-{}-{}-{}/weights".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second))
    remove_dir_and_create_dir("./logs/{}-{}-{}-{}-{}-{}/summary".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second))
    
    # run on gpu if cuda exists else cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on cuda")
    else:
        device = torch.device("cpu")
        print("Running on cpu")
    
    # check data for number of classes(categories)
    # You can change category path here
    category_path = "./data/traindata/categories.txt"
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
    
    

if __name__ == "__main__":
    main()