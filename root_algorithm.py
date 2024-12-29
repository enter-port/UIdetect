'''
The algorithm for calculatingm root
demo for single image
'''
from dataset.datasetGD import UIDataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

def main():

    # Note! change filename frame_31 to frame_31_29; frame_31 is based on frame_29
    # TODO: use this algoeithm iteraively in dataset; find the "previous" frame fo every frame and calculate its root
    # TODO: if one frame has no previous frame, the root is itself
    # img1 = cv2.imread('data/jiguang/frame_29.png')
    # img2 = cv2.imread('data/jiguang/frame_31_29.png')
    img1 = cv2.imread('data/jiguang/frame_1.png')
    img2 = cv2.imread('data/jiguang/frame_2_1.png')
    
    # check whether image shape are the same(should be the same in one category)
    if img1.shape != img2.shape:
        warnings.warn(f"Warning: The images have different sizes. img1 size: {img1.shape}, img2 size: {img2.shape}")
        return 

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # calculate pixel difference
    diff = cv2.absdiff(gray1, gray2)

    # turn to binary image
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # expansion
    kernel = np.ones((50, 50), np.uint8)  # kernel for expansion
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    # find the minimal contour
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter those contour far away from the biggest contour
    rects = []
    max_area = 0
    max_contour = None

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rects.append((x, y, w, h, area, contour))

            if area > max_area:
                max_area = area
                max_contour = contour

    if max_contour is None:
        print("No valid contour found.")
        return
    
    # find the center of the biggest contour
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cx_max = int(M["m10"] / M["m00"])
        cy_max = int(M["m01"] / M["m00"])
    else:
        cx_max, cy_max = 0, 0  

    # delete those contours far away
    filtered_rects = []
    max_distance = 0.5 * min(img1.shape[:2])

    for (x, y, w, h, area, contour) in rects:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        distance = np.sqrt((cx_max - cx)**2 + (cy_max - cy)**2)

        if distance < max_distance:
            filtered_rects.append((x, y, w, h))
            
    # for (x, y, w, h) in filtered_rects:
    #     cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    
    # combine all the bounding boxes into a big rectangle
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0    
    for (x, y, w, h) in filtered_rects:
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
        
    cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # # draw root bbox
    # for contour in contours:
    #     if cv2.contourArea(contour) > 500: 
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    
    # # combine all the bounding boxes into a big rectangle
    # x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0

    # for contour in contours:
    #     if cv2.contourArea(contour) > 1000: 
    #         x, y, w, h = cv2.boundingRect(contour)
    #         x_min = min(x_min, x)
    #         y_min = min(y_min, y)
    #         x_max = max(x_max, x + w)
    #         y_max = max(y_max, y + h)

    # cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    
if __name__ == '__main__':
    main()