#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import xml.etree.ElementTree as ET
import os


image_path = 'D:/Python/大作业/人工智能/self_dataset/60/images/val/'
path = 'D:/Python/大作业/人工智能/self_dataset/60/labels/val/'
def convert(size, box):
    # dw = 1. / size[0]
    # dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    # x = x * dw
    # w = w * dw
    # y = y * dh
    # h = h * dh
    return x, y, w, h


def convert_annotation(xml_filename):
    print(xml_filename)
    in_file = open(f'{path}{xml_filename}', encoding='UTF-8')
    image_id = xml_filename.split('.')[0]
    iw, ih = Image.open(f'{image_path}{image_id}.jpg').size
    out_file = open(f'{path}xywh_{image_id}.txt', 'w')  # 生成txt格式文件
    tree = ET.parse(source=in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        cls_id = 47
        if cls == 'apple':
            cls_id = 47
        elif cls == 'banana':
            cls_id = 46
        elif cls == 'orange':
            cls_id = 49
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((iw, ih), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# xml list
files = os.listdir(path)
for file in files:
    label_name, ind = file.split('.')
    #print(label_name)
    if ind == 'xml':
        convert_annotation(file)

