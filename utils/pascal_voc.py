import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg


class pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')   #data/pascal_voc/VOCdevkit
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')      #data/pascal_voc/VOCdevkit/VOC2007
        self.cache_path = cfg.CACHE_PATH             #data/pascal_voc/cache
        self.batch_size = cfg.BATCH_SIZE             #batch size 为 45
        self.image_size = cfg.IMAGE_SIZE             # image_size为 448*448
        self.cell_size = cfg.CELL_SIZE              #整张图片的cell_size*cell_size, 即7*7
        self.classes = cfg.CLASSES                  #pascal_voc的20个数据集
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))   #将类转化为索引字典
        self.flipped = cfg.FLIPPED      #true
        self.phase = phase             #传过来的是字符串"train"
        self.rebuild = rebuild         #False
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()    #call the function prepare, return gt_labels, 包含了经过水平翻转后的训练实例

    def get(self):   #获取到batch_size大小的图片和对应的label
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)  #读取图片
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']   #图片的label
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)  #随机打乱gt_labels
                self.cursor = 0
                self.epoch += 1     #epoch加1
        return images, labels      #返回batch_size大小的图片和对应的label，注意其shape

    def image_read(self, imname, flipped=False):   #读取图片
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))  #resize大小
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()  #call the function load_labels, return gt_labels
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')  #附加水平翻转训练实例
            gt_labels_cp = copy.deepcopy(gt_labels)  #deep copy
            for idx in range(len(gt_labels_cp)):  #循环一次处理一张图片
                gt_labels_cp[idx]['flipped'] = True # 原来的false变成true，即翻转
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels   #包含有经过水平翻转后的图片，包含经过水平翻转的训练实例

    def load_labels(self):    #加载标签文件，获取标签
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')    ##data/pascal_voc/cache/pascal_/train/_gt_labels.pkl

        if os.path.isfile(cache_file) and not self.rebuild:      #从cache文件中读取labels
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)    #data/pascal_voc/VOCdevkit/VOC2007

        if not os.path.exists(self.cache_path):    #如果不存在这个文件则创建
            os.makedirs(self.cache_path)

        if self.phase == 'train':       #train
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')   #加载trainval.txt文件
        else:                 #test
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]    #加载得到图片的索引，以供train或者test

        gt_labels = []   #
        for index in self.image_index: #得到train或者test文件中所有指向图片的index,
            label, num = self.load_pascal_annotation(index)   #读取annotation文件, return label and len(objs)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')  ##data/pascal_voc/VOCdevkit/VOC2007/JPEGImages/index.jpg
            gt_labels.append({'imname': imname,   #图片的路径
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:   #save the label to cache file
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):    # Load image and bounding boxes info from XML file in the PASCAL VOC format

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')  #data/pascal_voc/VOCdevkit/VOC2007/JPEGImages/index.jpg
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0] # 448所占图片高度的比例
        w_ratio = 1.0 * self.image_size / im.shape[1] # 448占图片宽度的比例
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))    #label数组维度 7*7*25， 一个cell只负责预测一个类别
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml') #data/pascal_voc/VOCdevkit/VOC2007/Annotations/index.xml
        tree = ET.parse(filename)   #解析xml文件
        objs = tree.findall('object')   #找到index指向的该xml文件中的所有object

        for obj in objs:     #记录出xml文件中object框的位置
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]  # class to index, matters
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]  # 将坐标(x1,y1,x2,y2)转变成(x_center,y_center,width,height)
            x_ind = int(boxes[0] * self.cell_size / self.image_size)   # 查看object的x_center落在哪个cell, 整张图片cell的数量为7*7
            y_ind = int(boxes[1] * self.cell_size / self.image_size)   # 查看object的y_center落在哪个cell， 整张图片cell的数量为7*7
            if label[y_ind, x_ind, 0] == 1:   #设置一个标记，看其是否被访问过,同时也表明这是个object
                continue
            label[y_ind, x_ind, 0] = 1   #设置标记，1是已经被访问过了,同时也表明这是个object
            label[y_ind, x_ind, 1:5] = boxes  #设置标记的框，框的形式为 ( x_center, y_center, width, height)
            label[y_ind, x_ind, 5 + cls_ind] = 1  #标记类别，pascal_voc数据集一共有20个类，哪个类是哪个，则在响应的位置上的index是1

        return label, len(objs)   #返回label，以及该index文件中object的数量
