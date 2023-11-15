from xml.dom.minidom import parse
import os
import cv2 as cv

def readxml(name):
    dom = parse(name)
    # 获取文档元素对象
    elem = dom.documentElement

    # 1. 获取size信息
    size_node = elem.getElementsByTagName("size")[0]
    width = size_node.getElementsByTagName("width")[0].childNodes[0].nodeValue
    height = size_node.getElementsByTagName("height")[0].childNodes[0].nodeValue
    depth = size_node.getElementsByTagName("depth")[0].childNodes[0].nodeValue

    size_info = (int(width), int(height), int(depth))
    # 2. 获取object信息
    objs_info = []
    obj_nodes = elem.getElementsByTagName("object")
    for obj in obj_nodes:
        obj_info = {}
        name = obj.getElementsByTagName("name")[0].childNodes[0].nodeValue
        obj_info['name'] = name
        bdbox = obj.getElementsByTagName("bndbox")[0]
        x_min = bdbox.getElementsByTagName("xmin")[0].childNodes[0].nodeValue
        x_max = bdbox.getElementsByTagName("xmax")[0].childNodes[0].nodeValue
        y_min = bdbox.getElementsByTagName("ymin")[0].childNodes[0].nodeValue
        y_max = bdbox.getElementsByTagName("xmax")[0].childNodes[0].nodeValue
        obj_info['bndbox'] = (int(x_min), int(x_max), int(y_min), int(y_max))
        objs_info.append(obj_info)

    return size_info, objs_info


def get_filelists(file_dir):
    list_directory = os.listdir(file_dir)
    filelists = []
    for file in list_directory:
        filelists.append(file)
    return filelists

def load_annotation(addr):
    filelist = get_filelists(addr)
    size_data = []
    obj_data = []
    for file in filelist:
        size_info, objs_info = readxml(addr + file)
        size_data.append(size_info)
        obj_data.append(objs_info)

    return size_data, obj_data
    # obj_data : obj_data[n][m]['bndbox'] = (x_min, x_max, y_min, y_max)

def generate(addr):
    """
    Generate train.txt val.txt test.txt
    :param addr: ROOT ADDR of the DataSet
    :return:
    """
    train_addr = addr + "train/"
    test_addr = addr + "test/"
    val_addr = addr + "val/"
    addrs = [(train_addr, "train"), (test_addr, "test"), (val_addr, "val")]
    for root , name in addrs:
        filelist = get_filelists(root)
        suffix = '.xml'
        fp = open(name + ".txt", 'w')
        if name != 'test':
            fp_trainval = open("../../ImageSets/Main/trainval.txt", 'w')
        if filelist[0].find('jpg')!= 0:
            suffix = '.jpg'
        for file in filelist:
            file = file.rstrip(suffix)
            fp.write(file)
            fp.write('\n')
            if name != 'test':
                fp_trainval.write(file)
                fp_trainval.write('\n')

if __name__ == '__main__':
    # change to your own addr
    addr = 'C:\\Users\\314\Desktop\dachuang\\2023practice\phase2\dataset\JPEGImages\\'
    generate(addr)
