## Reference

https://blog.csdn.net/weixin_42392454/article/details/125420887

https://github.com/Runist/torch_CenterNet

## 简介

这个branch是我们最后尝试整合的UI detect pipeline

将助教给的数据解压缩后放到`/data`文件夹下，并且将主目录的`categories.txt`也放入`/data`

`playground.py`没啥用处，是我尝试一些代码的草稿，随便删

## 安装

### Centernet

`requirments.txt`仅供参考。应该只需要pytorch和numpy，可以根据cuda版本选择pytorch版本，不一定与`requirments.txt`一致。

### GroundingDINO

请首先将https://github.com/Seeple/UIGD 复制到third_party目录下

然后遵照GroundingDINO的repo中的README完成环境配置

特别地，考虑到GroudingDINO中c++扩展的cuda部分编译很容易出错，这里在运行时默认将cpu-only设为True；如果电脑有GPU且编译没有出错可以自行更改

## 运行pipeline

### Inference on GroundingDINO

运行下述命令：

` python3 GD_inference.py`

这个命令会在GroundingDINO上inference dataset中所有的图片，并返回框并储存可视化后的图片

其中部分命令行参数，如box_threshold, nms等可以自行调整，里面default是我调的比较好的一组

也可以尝试将GD_demoprompt.txt中的内容复制到命令行中，这将运行`inference_on_a_image.py`，展示一张图片上的效果。


## TODO

1. 兼容原始的dataset和GroundingDINO dataset，GroundingDINO dataset由于只需要承担inference任务，所以只输出了图片，而且没有划分训练，测试集
2. root
3. level