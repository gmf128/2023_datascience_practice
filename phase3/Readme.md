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

`src\test.py`必须在训练后运行，否则缺少权重文件。或从此链接下载权重文件后放置在任意位置，并修改test.py中的文件读取权重的路径： https://box.nju.edu.cn/f/46484a6932cd4df5a30b/?dl=1 



**说明：**

yolov8在保存txt文件时，其格式与作业要求不符(yolo默认格式为：class_id, x, y, w, h, conf)，

因此，我对`ultralytics\engine\results.py`进行了修改：

```python
 def save_txt(self, txt_file, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            txt_file (str): txt file path.
            save_conf (bool): save confidence score or not.
        """
        boxes = self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f'{probs.data[j]:.2f} {self.names[j]}') for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, conf, d.xyxy[0][0], d.xyxy[0][3], d.xyxy[0][2], d.xyxy[0][1])# update
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(), )
                # line += (conf, ) * save_conf + (() if id is None else (id, ))
                texts.append(('%g ' * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, 'a') as f:
                f.writelines(text + '\n' for text in texts)
```

--------

## Acknowledgements



