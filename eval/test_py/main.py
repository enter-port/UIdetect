# -*- coding=utf-8 -*-
import csv
from test_py.count.count import *
from test_py.read_datas.read import *
from test_py.visualization.visualization import drawing


def TP_P_num_sum(csv_results, threshod):
    TP_num_sum = 0
    P_num_sum = 0
    threshod = float('%.4f' % threshod)
    if ERR_S == threshod and ERR_S != 0:
        err = open('error_analysis_{}_{}.csv'.format(CLASS_NAME, ERR_S), "w", newline='')
        visualization_FP_wrt = csv.writer(err)
        visualization_FP_wrt.writerow(["图片名", "坐标", "分数", "err_type"])
    for csv_result in csv_results:
        if '/' in csv_result[0]:
            img_name = csv_result[0].split('/')[-1]
        else:
            img_name = csv_result[0]
        # img_name = csv_result[0]
        img_coord_confidence_s = csv_result[1]
        # print(img_coord_confidence_s)
        img_coords = list()
        img_confidence = list()
        for item in img_coord_confidence_s:
            if item[4] > threshod:  # 测试得到的相似度
                img_coords.append(item[:-1])
                img_confidence.append(item[-1])
        # print(img_coords)
        P_num_sum += len(img_coords)
        xml_path_ = os.path.join(XML_ROOT, img_name[:-4] + '.xml')
        xml_coords = parse_xml(xml_path_, CLASS_NAME)
        # print(len(xml_coords))
        # 创建一个行为len(img_coords), 列为len(xml_coords)的数组
        if len(img_coords) == 0 or len(xml_coords) == 0:
            if len(xml_coords) == 0 and len(img_coords) > 0 and ERR_S == threshod and ERR_S != 0:
                for i in range(len(img_coords)):
                    w_fp = " ".join(map(str, img_coords[i]))
                    visualization_FP_wrt.writerow((img_name, w_fp, img_confidence[i], CLASS_NAME))
            if len(img_coords) == 0 and len(xml_coords) > 0 and ERR_S == threshod and ERR_S != 0:
                for i in range(len(xml_coords)):
                    w_fn = " ".join(map(str, xml_coords[i]))
                    visualization_FP_wrt.writerow((img_name, w_fn, '0', CLASS_NAME))
            continue

        iou_array = np.zeros((len(img_coords), len(xml_coords)))
        rows, cols = iou_array.shape
        confidence_array = np.zeros((rows, cols))
        for img_coords_index in range(rows):
            confidence_array[img_coords_index, :] = img_confidence[img_coords_index]
            for xml_coords_index in range(cols):
                iou_array[img_coords_index, xml_coords_index] = compute_iou(
                    img_coords[img_coords_index],
                    xml_coords[xml_coords_index])
                if iou_array[img_coords_index, xml_coords_index] < IOU_THRESH:
                    iou_array[img_coords_index, xml_coords_index] = 0
                    confidence_array[img_coords_index, xml_coords_index] = 0
        loop_num = 0

        visualization_FP = img_coords[:]
        Confidence_FP = img_confidence[:]
        visualization_FN = xml_coords[:]
        while (loop_num < rows and loop_num < cols):
            max_index_ = np.unravel_index(confidence_array.argmax(), confidence_array.shape)
            # print(max_index_)
            max_index_1 = np.unravel_index(iou_array[max_index_[0]].argmax(), iou_array[max_index_[0]].shape)
            max_index = (max_index_[0], max_index_1[0])
            # print(max_index)
            if confidence_array[max_index[0], max_index[1]] < 0.1:
                break
            confidence_array[max_index[0], :] = 0
            confidence_array[:, max_index[1]] = 0
            loop_num += 1
            # 误报的
            visualization_FP.remove(img_coords[max_index[0]])
            Confidence_FP.remove(img_confidence[max_index[0]])
            # 漏报的
            visualization_FN.remove(xml_coords[max_index[1]])

        # 写入误报的数据
        if ERR_S == threshod and ERR_S != 0 and len(visualization_FP) > 0:
            for i in range(len(visualization_FP)):
                w_fp = " ".join(map(str, visualization_FP[i]))
                visualization_FP_wrt.writerow((img_name, w_fp, Confidence_FP[i], CLASS_NAME))
        if ERR_S == threshod and ERR_S != 0 and len(visualization_FN) > 0:
            for i in range(len(visualization_FN)):
                w_fn = " ".join(map(str, visualization_FN[i]))
                visualization_FP_wrt.writerow((img_name, w_fn, '0', CLASS_NAME))

        # if threshod == ANALY_THRESH:
            # csv_wrt.writerow((img_name, len(xml_coords), len(img_coords), loop_num))
            # print img_name,len(xml_coords),len(img_coords),loop_num
        TP_num_sum += loop_num

    return TP_num_sum, P_num_sum


def run():
    # 计算所有图像group truth的bbox个数，由于存在某些存在group truth的图片，没有预测到任何bbox，所以召回率的分母
    # 需要从所有测试图片的xml文件中累加得到
    group_truth_num_sum = 0
    for parent, _, files in os.walk(IMG_ROOT):
        for file in files:
            xml_path = os.path.join(XML_ROOT, file[:-4] + '.xml')
            coords = parse_xml(xml_path, CLASS_NAME)
            group_truth_num_sum += len(coords)

    print("group_truth_num_sum:%d" % group_truth_num_sum)

    #
    precisions = list()
    recalls = list()
    ap = 0

    # 根据阈值，计算FP的累加数量 以及 预测框的累加个数
    csv_results = parse_csv(CSV_PATH, CLASS_NAME)
    with open(CLASS_NAME + str(IOU_THRESH) + ".csv", 'w', newline='') as t_file:  # newline=''
        csv_write = csv.writer(t_file)
        csv_write.writerow(("threshod", "precision", "recall", "TP_num_sum", "P_num_sum"))
        for threshod in np.linspace(0.1, 1.0, 181):  # 设置阈值个数
            print("threshod: %f " % threshod)
            TP_num_sum, P_num_sum = TP_P_num_sum(csv_results, threshod)
            print(TP_num_sum, P_num_sum)
            precision = float(TP_num_sum) / (P_num_sum + 1e-05)
            recall = float(TP_num_sum) / (group_truth_num_sum + 1e-05)
            print("precision: %f, recall: %f" % (precision, recall))
            row = (threshod, precision, recall, TP_num_sum, P_num_sum)
            precisions.append(precision)
            recalls.append(recall)
            csv_write.writerow(row)

        # 计算ap
        ap = voc_ap(np.array(recalls), np.array(precisions), False)
        print('*************')
        print("ap: %f" % ap)
        print('*************')
        ap = '%.3f' % ap
        csv_write.writerow(('ap:{}'.format(ap), '', ''))
        csv_write.writerow(('group_truth_num_sum:{}'.format(group_truth_num_sum), '', ''))
    drawing(recalls, precisions, CLASS_NAME, IOU_THRESH, ap)


if __name__ == '__main__':
    IOU_THRESH = 0.5                         # IOU
    ERR_S = 0.3                              # 可视化阈值下的错误图片坐标。为0不写，为0.3即分析该阈值下的
    IMG_ROOT = r'F:\mayun\map\test_pics'     # 图片路径
    XML_ROOT = r'F:\mayun\map\test_xml'      # 图片对应xml标注文件路径
    OTT_IMG_ROOT = r'F:\mayun\map\out_pics'  # 漏报误报错误可视化图片路径
    CSV_PATH = r'F:\mayun\map\out.csv'       # 算法输出结果： img_name, x_min y_min w h, confidence, CLASS_NAME
    CLASS_NAME = 'person'

    run()
    err_drawing(CLASS_NAME, ERR_S, IMG_ROOT, OTT_IMG_ROOT)
