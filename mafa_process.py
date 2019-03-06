# -*- coding: utf-8 -*-
import glob as gb
import os
import xml.etree.ElementTree as ET
from scipy.io import loadmat


def create_list_file():
    save_train_path = '/home/helinfei/PycharmProjects/FaceBoxes.Attention/data/MAFA_TRAIN/img_list.txt'
    save_test_path = '/home/helinfei/PycharmProjects/FaceBoxes.Attention/data/MAFA_TEST/img_list.txt'

    img_root = '/data/Face/detection/MAFA/images'

    with open(save_train_path, 'w') as f:
        train_list = gb.glob(os.path.join(img_root, 'train_*.jpg'))
        train_list.sort()
        train_name = [os.path.splitext(os.path.basename(path))[0] for path in train_list]
        f.writelines([name + '.jpg ' + name + '.xml' + '\n' for name in train_name])

    with open(save_test_path, 'w') as f:
        test_list = gb.glob(os.path.join(img_root, 'test_*.jpg'))
        test_list.sort()
        test_name = [os.path.splitext(os.path.basename(path))[0] + '\n' for path in test_list]
        f.writelines(test_name)


def create_xml_file():
    save_path = '/home/helinfei/PycharmProjects/FaceBoxes.Attention/data/MAFA_TRAIN/annotations'
    train_mat = '/data/Face/detection/MAFA/LabelTrainAll.mat'
    data = loadmat(train_mat)

    img_name = data['label_train']['imgName'].squeeze()
    label = data['label_train']['label'].squeeze()

    for i in range(len(img_name)):
        # 创建根节点
        anno = ET.Element("annotation")
        # 创建子节点，并添加属性
        folder = ET.SubElement(anno, "folder")
        folder.text = ''
        # 创建子节点，并添加数据
        filename = ET.SubElement(anno, "filename")
        filename.text = img_name[i][0]

        for j in range(len(label[i])):
            obj = ET.SubElement(anno, "object")
            name = ET.SubElement(obj, "name")
            name.text = 'face'
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = str(1)

            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(label[i][j, 0])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(label[i][j, 1])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(label[i][j, 0]) + int(label[i][j, 2]))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(label[i][j, 1]) + int(label[i][j, 3]))

        # 创建elementtree对象，写文件
        tree = ET.ElementTree(anno)
        file_path = os.path.join(save_path, os.path.splitext(img_name[i][0])[0] + '.xml')
        tree.write(file_path)


def cal_img_avgsize(imgs_folder):
    import glob as gb
    import cv2
    avg_h = 0.0
    avg_w = 0.0
    min_h = 100
    min_w = 100

    train_mat = '/data/Face/detection/MAFA/LabelTrainAll.mat'
    data = loadmat(train_mat)
    label = data['label_train']['label'].squeeze()
    for i in range(len(label)):
        for j in range(len(label[i])):
            w = label[i][j, 2]
            h = label[i][j, 3]
            min_h = min(min_h, h)
            min_w = min(min_w, w)
    img_list = gb.glob(os.path.join(imgs_folder, '*.jpg'))
    # for img_path in img_list:
    #     img = cv2.imread(img_path)
    #     h, w, _ = img.shape
    #     avg_h += h
    #     avg_w += w
    # avg_h /= len(img_list)
    # avg_w /= len(img_list)
    print('avg_h: {}, avg_w: {}, min_h: {}, min_w: {}'.format(avg_h, avg_w, min_h, min_w))


if __name__ == '__main__':
    # create_xml_file()
    cal_img_avgsize('/home/helinfei/PycharmProjects/FaceBoxes.Attention/data/MAFA_TRAIN/images')

