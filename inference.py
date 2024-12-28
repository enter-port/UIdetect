import cv2
import json
import torch
import numpy as np
from model.centernet import CenterNet
from utils.data_utils import parse_xml, cal_feature, image_resize

def main():
    torch.serialization.add_safe_globals([CenterNet])
    target = "class"
    input_shape = (1280, 1920)
    output_shape = (input_shape[0] // 4, input_shape[1] // 4)
    
    # TODO: load model from ./model_to_be_tested
    # An example as to how to load the model is given below
    model_path = "./logs/icon/centernet/weights/epoch=99_loss=0.9091_val_loss=14.4106.pt"
    num_cat = 4 if target == "class" else 3
    category =["clickable", "selectable", "scrollable", "disabled"] if target == "class" else ["level_0", "level_1", "level_2"]    
    model = torch.load(model_path, weights_only=False)
    
    # run on gpu if cuda exists else cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on cuda")
    else:
        device = torch.device("cpu")
        print("Running on cpu")
    
    # TODO: implement inferencing one image
    # You can put the following code in a loop or as a function to run on multiple images
    test_path = "./data/jiguang/frame_2_1"
    
    image = cv2.imread(test_path + ".png")
    with open(test_path + ".json", "r") as f:
        pboxes = json.load(f)
        
    coords, names = parse_xml(test_path + ".xml")
    new_names = []
    new_coords = []
    for coord, name in zip(coords, names):
        if name.lower() == "root":
            root_coords = coord
        elif name.lower() in category:
            new_coords.append(coord)
            new_names.append(name)
    names = new_names
    coords = new_coords
            
    x0, y0, x1, y1 = root_coords
    image = image[y0:y1, x0:x1]
    
    new_pboxes = []
    bboxes = []
    
    for i in range(len(coords)):
        bboxes.append([
            coords[i][0] - x0,
            coords[i][1] - y0,
            coords[i][2] - x0,
            coords[i][3] - y0
        ])
        
    for i in range(len(pboxes)):
        box = ([
            pboxes[i][0] - x0,
            pboxes[i][1] - y0,
            pboxes[i][2] - x0,
            pboxes[i][3] - y0
        ])
        if all(coord > 0 for coord in box):
            new_pboxes.append(box)
            
    pboxes = new_pboxes
    
    # resize
    image, bboxes, pboxes = image_resize(image, input_shape, np.array(bboxes), np.array(pboxes))
    
    # add indices
    category_indices = np.array([int(category.index(name.lower())) for name in names])
    bboxes = np.column_stack((bboxes, category_indices))
    pre_category_indices = np.array([0 for _ in range(len(pboxes))])
    pboxes = np.column_stack((pboxes, pre_category_indices))
    
    # gound truth
    gt_hm, gt_wh, gt_offset, gt_offset_mask = cal_feature(image, bboxes, output_shape, num_cat, 4)
    
    # pre_boxes from dino
    pre_hm, pre_wh, pre_offset, _ = cal_feature(image, pboxes, output_shape, 1, 4)
    
    # TODO: add batch dimension to all relavent tensors
    
    # TODO: get raw output from model: refer to ./train_icon line 202-215
    
    # TODO: postprocess output: refer to ./utils/train_utils get_image()
    
    # TODO: adjust how we draw the bounding box in ./utils/train_utils postprocess_output(). Originally in line 44-47 i added the logic "if elements in mask < 10 choose top five score and corresponding mask". TRY DIFFERENT STRAGEGIES HERE!
    
    # TODO: find out whether a rather high val_loss really means poor performance on test set.

if __name__ == '__main__':  
    main()