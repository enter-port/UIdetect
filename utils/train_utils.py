import torch
import cv2
import numpy as np
from torchvision.ops import nms
from utils.data_utils import recover_input
import shutil
import os

def postprocess_output(hms, whs, offsets, confidence, device):
    """
    The post process of model output.
    Args:
        hms: heatmap
        whs: the height and width of bounding box
        offsets: center point offset
        confidence: the threshold of heatmap
        dev: torch device

    Returns:  The list of bounding box(x, y, w, h, score, label).

    """
    batch, output_h, output_w, c = hms.shape

    detections = []
    for b in range(batch):
        # (h, w, c) -> (-1, c)
        heat_map = hms[b].view([-1, c])
        # (h, w, 2) -> (-1, 2)
        wh = whs[b].view([-1, 2])
        # (h, w, 2) -> (-1, 2)
        offset = offsets[b].view([-1, 2])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        xv, yv = xv.flatten().float(), yv.flatten().float()

        xv = xv.to(device)     # x axis coordinate of feature point
        yv = yv.to(device)     # y axis coordinate of feature point

        # torch.max[0] max value
        # torch.max[1] index of max value
        score, label = torch.max(heat_map, dim=-1)
        mask = score > confidence
        
        # here if elements in mask < 5 choose top five score and corresponding mask
        if mask.sum() < 5:
            topk_score, topk_index = torch.topk(score, 10)
            mask = torch.zeros_like(mask)
            mask[topk_index] = 1 

        # Choose height, width and offset by confidence mask
        wh_mask = wh[mask]
        offset_mask = offset[mask]

        if len(wh_mask) == 0:
            detections.append([])
            continue

        # Adjust center of predict box
        xv_mask = torch.unsqueeze(xv[mask] + offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + offset_mask[..., 1], -1)

        # Get the (xmin, ymin, xmax, ymax)
        half_w, half_h = wh_mask[..., 0:1] / 2, wh_mask[..., 1:2] / 2
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)

        # Bounding box coordinate normalize
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h

        # Concatenate the prediction
        detect = torch.cat(
            [bboxes, torch.unsqueeze(score[mask], -1), torch.unsqueeze(label[mask], -1).float()], dim=-1)
        detections.append(detect)

    return detections


def decode_bbox(prediction, input_shape, dev, image_shape=None, remove_pad=False, need_nms=False, nms_thres=0.4):
    """
    Decode postprocess_output output
    Args:
        prediction: postprecess_output output
        input_shape: model input shape
        dev: torch device
        image_shape: image shape
        remove_pad: model input is padding image, you should set remove_pad=True if you want to remove this pad
        need_nms: whether use NMS to remove redundant detect box
        nms_thres: nms threshold

    Returns:  The list of bounding box(x1, y1, x2, y2 score, label).

    """
    output = [[] for _ in prediction]

    for b, detection in enumerate(prediction):
        if len(detection) == 0:
            continue

        if need_nms:
            keep = nms(detection[:, :4], detection[:, 4], nms_thres)
            detection = detection[keep]

        output[b].append(detection)

        output[b] = torch.cat(output[b])
        if output[b] is not None:
            bboxes = output[b][:, 0:4]

            input_shape = torch.tensor(input_shape, device=dev)
            bboxes *= torch.cat([input_shape, input_shape], dim=-1)

            if remove_pad:
                assert image_shape is not None, \
                    "If remove_pad is True, image_shape must be set the shape of original image."
                ih, iw = input_shape
                h,  w = image_shape
                scale = min(iw/w, ih/h)
                nw, nh = int(scale * w), int(scale * h)
                dw, dh = (iw - nw) // 2, (ih - nh) // 2

                bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / scale
                bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / scale

            output[b][:, :4] = bboxes

    return output

def get_color_map():
    """
    Create color map.

    Returns: numpy array.

    """
    color_map = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            color_map[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return color_map

def draw_bbox(image, bboxes, labels, class_names, scores=None, show_name=False):
    """
    Draw bounding box in image.
    Args:
        image: image
        bboxes: coordinate of bounding box
        labels: the index of labels
        class_names: the names of class
        scores: bounding box confidence
        show_name: show class name if set true, otherwise show index of class

    Returns: draw result

    """
    color_map = get_color_map()
    image_height, image_width = image.shape[:2]
    draw_image = image.copy()

    for i, c in list(enumerate(labels)):
        bbox = bboxes[i]
        c = int(c)
        color = [int(j) for j in color_map[c]]
        if show_name:
            predicted_class = class_names[c]
        else:
            predicted_class = c

        if scores is None:
            text = '{}'.format(predicted_class)
        else:
            score = scores[i]
            text = '{} {:.2f}'.format(predicted_class, score)

        x1, y1, x2, y2 = bbox

        x1 = max(0, np.floor(x1).astype(np.int32))
        y1 = max(0, np.floor(y1).astype(np.int32))
        x2 = min(image_width, np.floor(x2).astype(np.int32))
        y2 = min(image_height, np.floor(y2).astype(np.int32))

        thickness = int((image_height + image_width) / (np.sqrt(image_height**2 + image_width**2)))
        fontScale = 0.35

        t_size = cv2.getTextSize(text, 0, fontScale, thickness=thickness * 2)[0]
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=color, thickness=thickness)
        cv2.rectangle(draw_image, (x1, y1), (x1 + t_size[0], y1 - t_size[1]), color, -1)  # filled
        cv2.putText(draw_image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale, (255, 255, 255), thickness//2, lineType=cv2.LINE_AA)

    return draw_image


def get_summary_image(images, input_shape, category, thresh,
                      hms_true, whs_true, offsets_true,
                      hms_pred, whs_pred, offsets_pred, device):
    summary_images = []
    
    outputs_true = postprocess_output(hms_true, whs_true, offsets_true, 0.999, device)
    outputs_true = decode_bbox(outputs_true, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    outputs_pred = postprocess_output(hms_pred, whs_pred, offsets_pred, thresh, device)
    outputs_pred = decode_bbox(outputs_pred, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    images = images.cpu().numpy()
    for i in range(len(images)):
        image = images[i]
        image = recover_input(image.copy())
        output_true = outputs_true[i]
        output_pred = outputs_pred[i]
        if len(output_true) != 0:
            output_true = output_true.data.cpu().numpy()
            labels_true = output_true[:, 5]
            bboxes_true = output_true[:, :4]
        else:
            labels_true = []
            bboxes_true = []
        
        if len(output_pred) != 0:
            output_pred = output_pred.data.cpu().numpy()
            labels_pred = output_pred[:, 5]
            bboxes_pred = output_pred[:, :4]
        else:
            labels_pred = []
            bboxes_pred = []
        
        image_true = draw_bbox(image, bboxes_true, labels_true, category, show_name=True)
        image_pred = draw_bbox(image, bboxes_pred, labels_pred, category, show_name=True)
        summary_images.append(np.hstack((image_true, image_pred)).astype(np.uint8))

    return summary_images

def get_image(images, input_shape, category, thresh, hms, whs, offsets, device):
    outputs = postprocess_output(hms, whs, offsets, thresh, device)
    outputs = decode_bbox(outputs, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    images = images.cpu().numpy()
    image = recover_input(images[0].copy())
    if len(outputs) != 0:
        outputs = outputs[0].data.cpu().numpy()
        labels = outputs[:, 5]
        bboxes = outputs[:, :4]
    else:
        labels = []
        bboxes = []

    return draw_bbox(image, bboxes, labels, category, show_name=True)

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
    