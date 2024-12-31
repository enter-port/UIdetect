'''
Inference on GroundingDINO
'''
import os
import json
import torch
import numpy as np

from utils.GD_utils import load_image, load_model, get_grounding_output

def GD_inference(dir, text_prompt = "UIBox . icon ."):
    '''
    Input: 
    dir: path to the original image
    text: text prompt input of Grounding DINO 
    '''
    config_file = "third_party/UIGD/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "third_party/UIGD/weights/groundingdino_swint_ogc.pth"
    box_threshold = 0.1
    text_threshold = 0.15
    
    # initialize the model
    model = load_model(config_file, checkpoint_path, cpu_only=True)
    
    # load image 
    image_path = dir + ".png"
    image_pil, image = load_image(image_path)
    
    # run model
    boxes_filt, _ = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, cpu_only = True, nms=True)
    
    # convert the result to xyxy and save in .json
    size = image_pil.size 
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
    }
    
    bboxes = []
    H, W = size[1], size[0]
    for box in boxes_filt:
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        bboxes.append([x0, y0, x1, y1])
    
    with open(dir + ".json", 'w') as f:
        json.dump(bboxes, f, indent=4)

def main():
    
    img_dir = "./data/coredraw/frame_1"
    GD_inference(img_dir)
    
    
if __name__ == '__main__':
    main()