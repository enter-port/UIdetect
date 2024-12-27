# 目标检测模型评估测试脚本

#### 介绍
评估目标检测模型指标，ap，recall，precision计算，漏报误报错误可视化
- AP: PR曲线下面积
- mAP: mean Average Precision, 即各类别AP的平均值
- recall：TP / (TP + FN)
- Precision: TP / (TP + FP)

#### 安装
pip3 install -r requirements.txt

#### 使用说明
- out_pics     显示漏报和误报的可视化结果
- test_pics    测试集图片
- test_xml     测试集图片标注
- test_py      测试脚本

###### 执行test_py中main.py 参数根据自己本地路径修改
- python3 test_py/mian.py
- out_pics 显示漏报和误报的可视化结果
- 计算结果csv等文件在mian.py同级目标生成