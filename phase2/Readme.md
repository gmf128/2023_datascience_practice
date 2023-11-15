# 项目2：漏油检测

## Structure

`dataset`存放数据集

`weights`存放backbone及训练保存的权重文件

`src`：code

`results`：对测试集的预测结果

## Install

### 1.安装依赖

### 2.数据集

请下载数据集并解压至根目录下的`dataset/`https://box.nju.edu.cn/f/72a1aa3875b84669a86f/?dl=1

路径应当如下：

`dataset`

----`Annotations`

--------`001.xml`...

----`ImageSets`

--------`Main`

------------`train.txt``test.txt`...

----`JPEGImages`

---------`001.jpg`

### 3.模型文件

下载`vgg16_reducedfc.pth`至根目录下的`weights/`:https://box.nju.edu.cn/f/0e040946336f4461b862/?dl=1

验证预训练模型，下载

至根目录下的`weights/`https://box.nju.edu.cn/f/30c54a9c260f4b93ab0c/?dl=1

--------

## Acknowledgements

SSD模型的开源实现：[amdegroot/ssd.pytorch: A PyTorch Implementation of Single Shot MultiBox Detector (github.com)](https://github.com/amdegroot/ssd.pytorch)

