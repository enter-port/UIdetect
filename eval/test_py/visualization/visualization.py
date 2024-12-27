import matplotlib.pyplot as plt
import matplotlib as mpl  # 在plt中正常显示中文字符
mpl.rcParams['font.sans-serif'] = ['SimHei']


def drawing(recalls, precisions, CLASS_NAME, IOU_THRESH, ap):
    # 画图
    plt.figure(figsize=(10, 8), dpi=100)  # 设置图片的大小和dpi
    # precisions.append(1.)   # 为了画图完整和避免除零错误，需要保存precision=1，recall=0的点
    # recalls.append(0.)
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')  # step填充画图
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    if len(precisions) != 0:
        plt.plot(recalls[0], precisions[0], 'ro')
        plt.plot(recalls[-1], precisions[-1], 'go')
        plt.text(recalls[0], precisions[0], (round(recalls[0], 3), round(precisions[0], 3)),
                 ha='center', va='bottom', fontsize=10)
        plt.text(recalls[-1], precisions[-1], (round(recalls[-1], 3), round(precisions[-1], 3)),
                 ha='center', va='bottom',
                 fontsize=10)
    plt.xlabel('Recall', fontsize=16)  # 设置坐标标签和大小
    plt.ylabel('Precision', fontsize=16)
    plt.ylim([0.0, 1.05])  # 设置坐标范围
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=16)  # 坐标记号
    plt.yticks(fontsize=16)
    plt.title('%s Precision-Recall curve: AP=%0.3f' % (CLASS_NAME, float(ap)), fontsize=20)
    plt.savefig(CLASS_NAME + str(IOU_THRESH) + '.png', dpi=300)  # 保存图片
    # plt.show()