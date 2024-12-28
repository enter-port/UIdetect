'''
Inference UI dataset on pretrained model of GroundingDINO
This file gives a modified version catering to our new pipeline
'''
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

from utils.GD_utils import plot_boxes_to_image_no_label, load_image_from_dataset, load_model, get_grounding_output
from dataset.datasetGD import UIDataset

def main():
    
    # add args from parser
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default="groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="weights/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.1, help="text threshold")
    parser.add_argument("--cpu-only", default=True, help="running on cpu only!, default=False")
    args = parser.parse_args()
    
    # intialize configs from args
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    
    # initialize the model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    
    # set the root directory to save all the predicted images(by system time)\
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    base_log_path = f"./logs/{timestamp}"

    # initialize dataloader
    category_path = "./data/categories.txt"
    input_shape = (1920, 1280)
    batch_size = 1
    # test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape) 
    test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape) 
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True)
    demo_dataset = test_dataset[:5]
    demo_loader = DataLoader(demo_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    # load data iteratively from dataloader, and return result dict
    tbar = tqdm(test_loader)
    # tbar = tqdm(demo_loader)
    res_list = []
    step = 1
    for images in tbar:
        '''
        (B * data_size, B=1 by default)
        '''
        for i in range(images.size()[0]):
            # the saving directory of each image should be under base_log_path
            image = images[i]
            text_prompt = 'ClickableUIBox . ScrollableUIBox . SelectableUIBox . Icon' # WE DONT CARE CATEGORIES IN NEW PIPELINE 
            
            # define output directory based on the basename of the image
            img_name = f'image_{step}'
            output_dir = os.path.join(base_log_path, img_name)  
            os.makedirs(output_dir, exist_ok=True)
            step += 1
            
            # load image
            image_pil, image = load_image_from_dataset(image)
            # save raw image
            image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
            
            # run model
            boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, nms=True)
            
            # visualize and save the pred
            size = image_pil.size 
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],  # H,W
            }
        
            image_with_box = plot_boxes_to_image_no_label(image_pil, pred_dict)[0]
            image_with_box.save(os.path.join(output_dir, "pred.jpg"))
            res_list.append(pred_dict)
            
    '''
    res_list is a list of the prediction results of each image
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    Here:
    boxes contain (center_x, center_y, width, height). 
    These values are normalized in [0, 1].
    denormalize it through "size"
    '''
        
    print('all the samples have been inferenced!')
    
if __name__ == '__main__':
    main()