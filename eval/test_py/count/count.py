import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.
        # 2010年以前按recall等间隔取11个不同点处的精度值做平均(0., 0.1, 0.2, …, 0.9, 1.0)
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                # 取最大值等价于2010以后先计算包络线的操作，保证precise非减
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # 2010年以后取所有不同的recall对应的点处的精度值做平均
        # first append sentinel values at the end
        # mrec = np.concatenate(([0.], rec, [1.]))
        # mpre = np.concatenate(([0.], prec, [0.]))
        mrec = np.concatenate(([0.], rec[::-1], [1.]))
        mpre = np.concatenate(([0.], prec[::-1], [0.]))
        # 计算包络线，从后往前取最大保证precise非减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 找出所有检测结果中recall不同的点
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        # 用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (
                S1 + S2 - S_cross + 1e-05)  # we add small epsilon of 1e-05 to avoid division by 0