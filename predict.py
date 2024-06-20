import math
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from ultralytics import YOLO


def calculate_iou(box1, box2): # x y w h
    xmin1 = box1[0] - box1[2]/2
    xmin2 = box2[0] - box1[2]/2
    xmax1 = box1[0] + box1[2]/2
    xmax2 = box2[0] + box1[2]/2
    ymin1 = box1[1] - box1[3]/2
    ymin2 = box2[1] - box1[3]/2
    ymax1 = box1[1] + box1[3]/2
    ymax2 = box2[1] + box1[3]/2
    inner = max(0, min(xmax1, xmax2) - max(xmin1, xmin2)) * max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - inner
    # 计算iou
    iou = inner / union

    #print('iou_:', iou)
    return iou


def iou(boxs1, boxs2):
    sum = 0
    for box1 in boxs1:
        m = []
        for box2 in boxs2:
            m.append(calculate_iou(box1, box2))
        mm = max(m)
        sum += mm
    iou = sum/len(boxs1)
    return iou


image_path = 'D:/Python/大作业/人工智能/self_dataset/60/images/val/'
path = 'D:/Python/大作业/人工智能/self_dataset/60/labels/val/'


def convert(box):
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x, y, w, h


def convert_annotation(xml_filename, id, allAllow = False):
    # print('id', id)
    in_file = open(f'{path}{xml_filename}', encoding='UTF-8')
    tree = ET.parse(source=in_file)
    root = tree.getroot()
    bb = []
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
        if id == cls_id or allAllow:
            bb.append(convert(b))
    return torch.tensor(bb)


def calIou(path):
    files = os.listdir(path)
    mix, orange, banana, apple = 0, 0, 0, 0
    count_m, count_o, count_b, count_a = 0, 0, 0, 0
    for file in files:
        label_name, ind = file.split('.')
        # print(label_name)
        kind = label_name.split('_')[0]
        if ind == 'jpg' or ind == 'png':
            img_path = f'{path}{file}'
            results = model.predict(img_path, conf=0.5,
                                    device='cuda:0',
                                    classes=[46, 47, 49])

            for i, r in enumerate(results):
                # Plot results image
                # print('>>>>>>>>>>>>>>>>>>', r.boxes)
                allAllow = False
                id = 0
                if kind == 'apple':
                    id = 47
                elif kind == 'orange':
                    id = 49
                elif kind == 'banana':
                    id = 46
                elif kind == 'mixed':
                    allAllow = True

                xywh1 = r.boxes.xywh
                xywh2 = convert_annotation(f'{label_name}.xml', id, allAllow)
                if kind == 'apple':
                    count_a += 1
                    apple += iou(xywh1, xywh2)
                elif kind == 'orange':
                    count_o += 1
                    orange += iou(xywh1, xywh2)
                elif kind == 'banana':
                    count_b += 1
                    banana += iou(xywh1, xywh2)
                elif kind == 'mixed':
                    count_m += 1
                    mix += iou(xywh1, xywh2)
    return apple / (count_a + 1e-7), orange / (count_o + 1e-7), banana / (count_b + 1e-7), mix / (count_m + 1e-7),(apple + orange + banana + mix) / (count_a + count_o + count_b + count_m + 1e-7)



def pred(image_path):
    results = model.predict(image_path, conf=0.5,
                            device='cuda:0',
                            classes=[46, 47, 49])
    # Display results
    name = image_path.split('/')[-1]
    for i, r in enumerate(results):
        # Plot results image
        # print(r.boxes)
        im_bgr = r.plot(conf=True, save=True)  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        # Show results to screen (in supported environments)
        im_rgb.save(f'D:/Python/大作业/人工智能/pre/{name}')
        # im_rgb.show()
        # r.show()


def predDir(path):
    files = os.listdir(path)
    for file in files:
        label_name, ind = file.split('.')
        # print(label_name)
        if ind == 'jpg' or ind == 'png' or ind == 'jpeg':
            pred(f'{path}{file}')


if __name__ == '__main__':
    # model = YOLO('D:/Python/大作业/人工智能/yolov8m.pt')
    model = YOLO('D:/Python/大作业/人工智能/runs/detect/train3/weights/best.pt')
    # model = YOLO('D:/Python/大作业/人工智能/yolov8x.pt')
    a,o,b,m,all = calIou('D:/Python/大作业/人工智能/self_dataset/60/images/val/')
    metrics = model.val(data="D:/Python/大作业/人工智能/self_dataset/60/60.yaml", plots=True, device='cuda:0',save_json=True)
    print('apple_iou:', a)
    print('orange_iou:', o)
    print('banana_iou:', b)
    print('mix_iou:', m)
    print('all_iou', all)
    # metrics = model.train(data="D:/Python/大作业/人工智能/self_dataset/60/60.yaml", plots=True, device='cuda:0', batch=4, epochs=16)
    # pred('D:/Python/大作业/人工智能/photos/bg1.jpg')
    # predDir('D:/Python/大作业/人工智能/photos/')
    # pred('D:/Python/大作业/人工智能/photos/bg2.jpg')
