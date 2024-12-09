## Reference

https://blog.csdn.net/weixin_42392454/article/details/125420887

https://github.com/Runist/torch_CenterNet

## 简介

尝试使用CenterNet进行UI按键检测

将助教给的数据解压缩后放到`/data`文件夹下，并且将主目录的`categories.txt`也放入`/data`

`playground.py`没啥用处，是我尝试一些代码的草稿，随便删

## 环境

`requirments.txt`仅供参考。应该只需要pytorch和numpy，可以根据cuda版本选择pytorch版本，不一定与`requirments.txt`一致。

## 目前已完成的

1. 适配数据集的`dataset.py`文件，包括了`heatmap`的实现，给出的数据与https://github.com/Runist/torch_CenterNet中实现的一致。

## TODO

1. 实现training：参考CenterNet的写法