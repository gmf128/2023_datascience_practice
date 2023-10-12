# 项目1：杆塔检测

## Structure

`dataset`存放数据集

`weights`存放backbone及训练保存的权重文件

`src`：code

`results`：对测试集的预测结果

## Install

### 1.安装依赖

```
pip install -r requirements.txt
```

### 2.数据集

请下载数据集并解压至根目录下的`dataset/`https://box.nju.edu.cn/f/3537f4b7dfd148f58592/?dl=1

路径应当如下：

`dataset`

----`Annotations`

--------`0001.xml`...

----`ImageSets`

--------`Main`

------------`train.txt``test.txt`...

----`JPEGImages`

---------`0001.jpg`

### 3.模型文件

下载`vgg16_reducedfc.pth`至根目录下的`weights/`:https://box.nju.edu.cn/f/0e040946336f4461b862/?dl=1

验证预训练模型，下载https://box.nju.edu.cn/f/febe3f8224a44cc6ad3a/?dl=1

至根目录下的`weights/`

--------

## Acknowledgements

SSD模型的开源实现：[amdegroot/ssd.pytorch: A PyTorch Implementation of Single Shot MultiBox Detector (github.com)](https://github.com/amdegroot/ssd.pytorch)

