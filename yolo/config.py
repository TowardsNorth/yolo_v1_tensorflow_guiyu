import os

#
# path and dataset parameter
# 配置文件
#

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')  # 存放输出文件的地方，data/pascal_voc/output

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')  # weights_dir, 路径为data/pascal_voc/weights

WEIGHTS_FILE = None  # weights file
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']  # PASCAL VOC数据集的20个类别

FLIPPED = True

#
# model parameter
#

IMAGE_SIZE = 448  # 输入图片的大小

CELL_SIZE = 7  # 整张图片分为cell_size * cell_size的大小

BOXES_PER_CELL = 2  # 每个cell负责预测两个bounding box

ALPHA = 0.1  #

DISP_CONSOLE = False

# 下面这几个是论文中涉及的参数
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001  # 学习率

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 45  # batch size

MAX_ITER = 1  # 迭代次数，可自定义

SUMMARY_ITER = 10

SAVE_ITER = 1000

#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5  # IOU阈值0.5
