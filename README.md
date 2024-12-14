## Reference

https://blog.csdn.net/weixin_42392454/article/details/125420887

https://github.com/Runist/torch_CenterNet

## 简介

尝试使用多种目标检测模型进行UI按键检测，目前已经尝试了CenterNet和DINO，后面打算尝试Grounding DINO

由于数据量小，打算分别尝试learn-from-scratch和fine-tune-on-pretrained model的做法，（据说GroudingDINO zero-shot 也不错）

`playground.py`没啥用处，是我尝试一些代码的草稿，随便删

## 安装

### 数据准备

将助教给的数据解压缩后放到`/data`文件夹下，并且将主目录的`categories.txt`也放入`/data`


### 环境配置

#### 复制本仓库到本地

`git clone https://github.com/enter-port/UIdetect.git` 
(如果是本项目的合作者，建议将https连接转为ssh连接)

建议创建虚拟环境，这里我直接使用的python venv，推荐python版本3.10

#### 复制DINO仓库

创建目录`/third_party`,将我们fork的DINO仓库下载到该目录下
`git clone https://github.com/enter-port/DINO-UI.git `，并将目录名改为DINO_UI

注：我们基于原先的DINO项目进行了一些细微的模型修改以适应我们的项目，所以请务必从我们fork的repo上复制

#### 安装依赖

1、`requirments.txt`仅供参考。应该只需要pytorch和numpy，可以根据cuda版本选择pytorch版本，不一定与`requirments.txt`一致
(但为了方便也可以直接`pip install -r requirements.txt`)

2、安装DINO所需要的依赖：先进入DINO的根目录，再DINO项目的按照要求安装依赖，以及编译CUDA

#### 环境变量配置

1、若完全按照上述流程配置本地仓库，为了正常运行train_dino.py,需设置如下环境变量（将`root\path`替换位实机的路径）：
` export PYTHONPATH=/root/path/UIdetect/third_party/DINO/models/dino:$PYTHONPATH  `
` export PYTHONPATH=/root/path/UIdetect/third_party/DINO:$PYTHONPATH  `
（只要将DINO中模型调用文件的父目录放入PYTHONPATH即可）

## 运行

### Centernet

#### learn from scratch

#### fine-tune

### DINO

#### learn_from_sccratch
` python3 train_dino.py`
#### fine-tune
fine-tune还没写

## 目前已完成的

1. Centernet的数据集配置适配数据集的`dataset.py`文件，包括了`heatmap`的实现，给出的数据与https://github.com/Runist/torch_CenterNet中实现的一致，以及训练和可视化流程

2. DINO learn from scratch

## TODO

1. DINO结果转化为`.xml`
2. DINO的训练（手上没卡，目前只跑了一个epoch，目前来看数据有点怪，可能需要陈神帮忙后续debug一下train和eval的部分）
3. 运行`eval`得出结果，使用eval测试现有的两个模型
4. Grounding DINO测试zero-shot能力
5. 对于Centernet和DINO finetune（目前我能想到的就是在pre-trained model上面用我们的数据集继续训，改一下训练超参，也不用特别复杂）
6. 尝试数据增强（Centernet自带的数据增强）