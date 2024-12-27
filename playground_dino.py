from dataset.dino_dataset import UIDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2

def deprocess_input(image):
    # 逆标准化：从 (C, H, W) 到 (H, W, C)，并恢复原始的像素值
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]

    # 先转换回 (H, W, 3)
    image = np.transpose(image, (1, 2, 0))

    # 逆归一化：image * std + mean
    image = image * std + mean  # 恢复到[0, 1]
    
    # 逆标准化：恢复到[0, 255]之间的像素值
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)  # 确保在 [0, 255] 范围内并转换为 uint8 类型

    return image

def visualize_and_save(gt_info, res_info, image_name, image, save_path, original_size):
    """
    可视化并保存图像，绘制预测框与真实框（gt_info为归一化坐标）。

    :param gt_info: 真实框信息 (Tensor) [x_min, y_min, x_max, y_max, label] (归一化坐标)
    :param res_info: 预测框信息 (Tensor) [x_min, y_min, x_max, y_max, score, label]
    :param image_name: 图像名称
    :param image: 图像数据 (ndarray, RGB格式, CHW)
    :param save_path: 保存路径
    :param original_size: 原始图像大小 (width, height)
    """

    # 将图像从 CHW 格式转换为 HWC 格式，并从 RGB 转换为 BGR (OpenCV 使用 BGR)
    image_bgr = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    image_bgr = image_bgr[:, :, ::-1]  # RGB -> BGR

    # 恢复到原始尺寸
    image_bgr = cv2.resize(image_bgr, (original_size[0], original_size[1]))

    # 获取原始图像的宽度和高度
    img_width, img_height = original_size

    # 画真实框 (gt_info), 这里的gt_info是归一化坐标，转换为像素坐标
    for box in gt_info:
        # 从归一化坐标转换为像素坐标
        x_min, y_min, x_max, y_max, label = box.tolist()
        x_min = int(x_min * img_width)
        y_min = int(y_min * img_height)
        x_max = int(x_max * img_width)
        y_max = int(y_max * img_height)
        
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image_bgr, f"GT: {int(label)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 画预测框 (res_info)
    for box in res_info:
        x_min, y_min, x_max, y_max, score, label = box.tolist()
        color = (0, 0, 255)  # 红色
        cv2.rectangle(image_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(image_bgr, f"Pred: {int(label)}: {score:.2f}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示图像
    cv2.imshow(f"Image: {image_name}", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存图像
    cv2.imwrite(save_path, image_bgr)

def main():
    input_shape = (1280, 1920)
    category_path = "./data/categories.txt"
    
    test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape, is_train=False) 
    
    image, target = test_dataset[0] # image: (3,1280, 1960), np.ndarray
    image_restored = deprocess_input(image)
    image_bgr = image_restored[:, :, ::-1]
    print("target data:",target)
    
    cv2.imshow("Demo Image", image_bgr) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()