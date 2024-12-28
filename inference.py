import cv2
import json
import torch
import numpy as np
from model.centernet import CenterNet
from dataset.dataset import UIDataset
from torch.utils.data import DataLoader
from utils.data_utils import parse_xml, cal_feature, image_resize
from utils.train_utils import postprocess_output, decode_bbox, draw_bbox
from utils.data_utils import recover_input, preprocess_input

def infer_on_an_image(model, img_path, category, 
                      input_shape, output_shape, 
                      num_cat, device, vis = True):
    image = cv2.imread(img_path + ".png")
    
    with open(img_path + ".json", "r") as f:
        pboxes = json.load(f)
        
    coords, names = parse_xml(img_path + ".xml")
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
    image = preprocess_input(image)
    
    # add indices
    category_indices = np.array([int(category.index(name.lower())) for name in names])
    bboxes = np.column_stack((bboxes, category_indices))
    pre_category_indices = np.array([0 for _ in range(len(pboxes))])
    pboxes = np.column_stack((pboxes, pre_category_indices))
    
    # gound truth
    gt_hm, gt_wh, gt_offset, gt_offset_mask = cal_feature(image, bboxes, output_shape, num_cat, 4)
    
    # pre_boxes from dino
    pre_hm, pre_wh, pre_offset, _ = cal_feature(image, pboxes, output_shape, 1, 4)
    
    # Convert to tensors and add batch dimension (B=1)
    images = torch.from_numpy(image).float().unsqueeze(0).to(device)  # CHW -> BCHW
    # print(images.shape)
    gt_hms = torch.from_numpy(gt_hm).float().unsqueeze(0).to(device)  
    gt_whs = torch.from_numpy(gt_wh).float().unsqueeze(0).to(device)
    gt_offsets = torch.from_numpy(gt_offset).float().unsqueeze(0).to(device) 
    gt_offset_masks = torch.from_numpy(gt_offset_mask).float().unsqueeze(0).to(device) 
    pre_hms = torch.from_numpy(pre_hm).float().unsqueeze(0).to(device)  
    pre_whs = torch.from_numpy(pre_wh).float().unsqueeze(0).to(device)  
    pre_offsets = torch.from_numpy(pre_offset).float().unsqueeze(0).to(device) 
    
    inference_output = model(images, mode='train', 
                                    pre_box_data = (pre_hms, pre_whs, pre_offsets),
                                    ground_truth_data=(gt_hms, gt_whs, gt_offsets, gt_offset_masks)
                            )
    # print("inference output", inference_output)
    hms_pred, whs_pred, offsets_pred, loss, c_loss, wh_loss, off_loss, hms_true = inference_output
    
    # average the loss in a batch 
    loss = loss.mean()
    c_loss = c_loss.mean()
    wh_loss = wh_loss.mean()
    off_loss = off_loss.mean()
    
    thresh = 0.2
    
    outputs = postprocess_output(hms_pred, whs_pred, offsets_pred, thresh, device)
    outputs = decode_bbox(outputs, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    outputs_true = postprocess_output(gt_hms, gt_whs, gt_offsets, 0.999, device)
    outputs_true = decode_bbox(outputs_true, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    
    if vis:
        images = images.cpu().numpy()
        image = recover_input(images[0].copy())
        # cv2.imshow("image after recover", image)
        if len(outputs) != 0:
            outputs = outputs[0].data.cpu().numpy()
            labels = outputs[:, 5]
            bboxes = outputs[:, :4]
        else:
            labels = []
            bboxes = []

        return draw_bbox(image, bboxes, labels, category, show_name=True), loss, c_loss, wh_loss, off_loss
    else:
        return outputs[0].data, loss, c_loss, wh_loss, off_loss
    
    # TODO: adjust how we draw the bounding box in ./utils/train_utils postprocess_output(). Originally in line 44-47 i added the logic "if elements in mask < 10 choose top five score and corresponding mask". TRY DIFFERENT STRAGEGIES HERE!
    
    # TODO: find out whether a rather high val_loss really means poor performance on test set.(Likely to be no. WHY?)

def main():
    # torch.serialization.add_safe_globals([CenterNet])
    target = "class"
    input_shape = (1280, 1920)
    output_shape = (input_shape[0] // 4, input_shape[1] // 4)
    
    # TODO: load model from ./model_to_be_tested
    # model_path = "./model_to_be_tested/icon/model1/epoch=145_loss=5.0250_val_loss=13.1132.pt"
    model_path = "./model_to_be_tested/icon/model1/epoch=599_loss=0.0515_val_loss=20.6104.pt"
    num_cat = 4 if target == "class" else 3
    category =["clickable", "selectable", "scrollable", "disabled"] if target == "class" else ["level_0", "level_1", "level_2"]    
    model = torch.load(model_path, weights_only=False)
    model.eval()
    
    # run on gpu if cuda exists else cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on cuda")
    else:
        device = torch.device("cpu")
        print("Running on cpu")
    
    # TODO: implement inferencing one image
    # iteratively get images from eval_datasets 
    total_loss = []
    total_c_loss = []
    total_wh_loss = []
    total_offset_loss = []
    vis = True
    
    # # build data loader
    # data_path = "./data"
    # batch_size = 1
    # test_dataset = UIDataset(data_path, input_shape=input_shape, is_train=False, target=target)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # test on single image
    # try a image in test set
    # img_path = "./data/coredraw/frame_18_17"
    # try a image in train set
    img_path = "./data/coredraw/frame_8_5"
    image = cv2.imread(img_path + ".png")
    # cv2.imshow("original image", image)
    pred_img, loss, c_loss, wh_loss, off_loss = infer_on_an_image(model, img_path, category, input_shape, output_shape, num_cat, device, vis)
    
    cv2.imshow("image with box", pred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':  
    main()