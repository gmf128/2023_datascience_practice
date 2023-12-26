# 项目2：漏油检测

## Structure

`datasets`存放数据集

`src`：code

`cfg`：配置文件

## Install

### 1.安装依赖

```
# 从PyPI安装ultralytics包
pip install ultralytics
```

### 2.数据集

请下载数据集并解压至根目录下的`datasets/`

路径应当如下：

`datasets`

----`labels`

---------`val`

---------`train`

----`images`

---------`test`

---------`val`

---------`train`

### 3.运行

`src\train.py`训练，第一次运行时会下载预训练权重文件

`src\test.py`必须在训练后运行，否则缺少权重文件。

--------

## Acknowledgements



