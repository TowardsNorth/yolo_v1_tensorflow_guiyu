import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim


#定义YoloNet网络
class YOLONet(object):

    def __init__(self, is_training=True):    #类的初始化
        self.classes = cfg.CLASSES    #PASCAL VOC数据集的20个数据类
        self.num_class = len(self.classes)  #20个类别
        self.image_size = cfg.IMAGE_SIZE   #图片的大小
        self.cell_size = cfg.CELL_SIZE    #整张输入图片划分为cell_size * cell_size的网格
        self.boxes_per_cell = cfg.BOXES_PER_CELL  #每个cell负责预测多少个(mayebe 2)bounding box
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)    #最后输出的tensor大小，其为S*S*(C+5*B),具体可以看论文
        self.scale = 1.0 * self.image_size / self.cell_size  #每个cell像素的大小
        self.boundary1 = self.cell_size * self.cell_size * self.num_class  #类似于7*7*20
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell  #类似于 7*7*20 + 7*7*2

        self.object_scale = cfg.OBJECT_SCALE   #这些是论文中涉及的参数，具体可看论文(You Only Look Once: Unified, Real-Time Object Detection)
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE  #学习率
        self.batch_size = cfg.BATCH_SIZE  #batch_size
        self.alpha = cfg.ALPHA  #

        self.offset = np.transpose(np.reshape(np.array(   #reshape之后再转置，变成7*7*2的三维数组
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))



        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')   #定义输入的placeholder，需要喂饱的数据, batch_size * 448 * 448 *3 ,
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)  # 构建网络，预测值，在本程序中，其格式为 [batch_size , 7 * 7 * （20 + 2 * 5）]，其中的20表示PASCAL VOC数据集的20个类别

        if is_training:   #training为true时
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])   # 需要喂饱的数据，在本程序中其格式为：[batch_size,7 ,7 ,25]
            self.loss_layer(self.logits, self.labels)   #预测值和真实值的比较，得到loss
            self.total_loss = tf.losses.get_total_loss()   #将所有的loss求和
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,     #用slim构建网络，简单高效
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],  #卷积层加上全连接层
                activation_fn=leaky_relu(alpha),   #用的是leaky_relu激活函数
                weights_regularizer=slim.l2_regularizer(0.0005), #L2正则化，防止过拟合
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)  #权重初始化
            ):

                #这里先执行填充操作
                # t = [[2, 3, 4], [5, 6, 7]], paddings = [[1, 1], [2, 2]]，mode = "CONSTANT"
                #
                # 那么sess.run(tf.pad(t, paddings, "CONSTANT"))
                # 的输出结果为：
                #
                # array([[0, 0, 0, 0, 0, 0, 0],
                #        [0, 0, 2, 3, 4, 0, 0],
                #        [0, 0, 5, 6, 7, 0, 0],
                #        [0, 0, 0, 0, 0, 0, 0]], dtype=int32)
                #
                # 可以看到，上，下，左，右分别填充了1, 1, 2, 2
                # 行刚好和paddings = [[1, 1], [2, 2]]
                # 相等，零填充
                #因为这里有4维，batch和channel维没有填充，只填充了image_height,image_width这两个维度，0填充
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')  #这里的64是指卷积核个数，7是指卷积核的高度和宽度，2是指步长，valid表示没有填充
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')  #max_pool, 大小2*2, stride:2
                net = slim.conv2d(net, 192, 3, scope='conv_4')   #这里的192是指卷积核的个数，3是指卷积核的高度和宽度，默认的步长为1
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')  #max_pool,大小为2*2，strides:2
                net = slim.conv2d(net, 128, 1, scope='conv_6') #128个卷积核，大小为1*1，默认步长为1
                net = slim.conv2d(net, 256, 3, scope='conv_7') #256个卷积核，大小为3*3，默认步长为1
                net = slim.conv2d(net, 256, 1, scope='conv_8') #256个卷积核，大小为1*1，默认步长为1
                net = slim.conv2d(net, 512, 3, scope='conv_9') #512个卷积核，大小为3*3，默认步长为3
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10') #max_pool, 大小为2*2，stride:2
                net = slim.conv2d(net, 256, 1, scope='conv_11')  #256个卷积核，大小为1*1, 默认步长为1
                net = slim.conv2d(net, 512, 3, scope='conv_12')  #512个卷积核，大小为3*3,默认步长为1
                net = slim.conv2d(net, 256, 1, scope='conv_13')  #256个卷积核，大小为1*1, 默认步长为1
                net = slim.conv2d(net, 512, 3, scope='conv_14')   #512个卷积核，大小为3*3, 默认步长为1
                net = slim.conv2d(net, 256, 1, scope='conv_15')  #256个卷积核，大小为1*1, 默认步长为1
                net = slim.conv2d(net, 512, 3, scope='conv_16')  #512个卷积核，大小为3*3, 默认步长为1
                net = slim.conv2d(net, 256, 1, scope='conv_17')  #256个卷积核，大小为1*1, 默认步长为1
                net = slim.conv2d(net, 512, 3, scope='conv_18')   #512个卷积核，大小为3*3, 默认步长为1
                net = slim.conv2d(net, 512, 1, scope='conv_19')  #256个卷积核，大小为1*1, 默认步长为1
                net = slim.conv2d(net, 1024, 3, scope='conv_20')  #1024个卷积核，大小为3*3，默认步长为1
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21') # max_pool, 大小为2*2，strides: 2
                net = slim.conv2d(net, 512, 1, scope='conv_22')  #512卷积核，大小为1*1，默认步长为1
                net = slim.conv2d(net, 1024, 3, scope='conv_23') #1024卷积核，大小为3*3，默认步长1
                net = slim.conv2d(net, 512, 1, scope='conv_24')  #512卷积核，大小为1*1，默认步长1
                net = slim.conv2d(net, 1024, 3, scope='conv_25')  #1024卷积核，大小为3*3, 默认步长为1
                net = slim.conv2d(net, 1024, 3, scope='conv_26')  #1024卷积核，大小为3*3，默认步长为1
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')     #padding, 第一个维度batch和第四个维度channels不用管，只padding卷积核的高度和宽度
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')  #1024卷积核，大小3*3，步长为2
                net = slim.conv2d(net, 1024, 3, scope='conv_29')   #1024卷积核，大小为3*3，默认步长为1
                net = slim.conv2d(net, 1024, 3, scope='conv_30')   #1024卷积核，大小为3*3，默认步长为1
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31') #转置，由[batch, image_height,image_width,channels]变成[bacth, channels, image_height,image_width]
                net = slim.flatten(net, scope='flat_32')  #将输入扁平化，但保留batch_size, 假设第一位是batch，实际上第一维也是batch
                net = slim.fully_connected(net, 512, scope='fc_33')   #全连接层,神经元个数
                net = slim.fully_connected(net, 4096, scope='fc_34')  #全连接层，神经元个数
                net = slim.dropout(  #dropout，防止过拟合
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(    #全连接层
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):   #计算IOU的函数
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    #这个是定义损失函数的，具体可以看论文
    def loss_layer(self, predicts, labels, scope='loss_layer'):   #定义损失函数，损失函数的具体形似可以查看论文, label的格式为[batch_size, 7, 7, 25]
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(  #reshape一下，每个cell一个框，变成[batch_size, 7, 7, 20]
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape( #reshape一下，7*7*20 ~ 7*7*22, 就是分别找到每个cell的两个框的置信度,这里是两个框，可自定义,变成[batch_size, 7, 7, 2]
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape( #reshape，就是分别找到每个cell中两个框的坐标（x_center, y_center, w, h），这里是两个框，可自定义, 变成[batch_size, 7, 7, 2, 4]
                predicts[:, self.boundary2:],  #7 * 7 * 22 ~ 7 * 7 * 30，
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            #下面是对label部分进行reshape
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])    #reshape, 就是查看哪个cell负责标记object,是的话就为1 ，否则是0 ，维度形式：[batch_size, 7, 7, 1]
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])  #找到这个cell负责的框的位置，其形式为：(x_center,y_center,width,height), 其维度为：[batch_size, 7, 7, 1, 4]
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size   # tile() 平铺之意，用于在同一维度上的复制, 变成[batch_size, 7, 7, 2, 4]， 除以image_size就是得到相对于整张图片的比例
            classes = labels[..., 5:]          #找到这个cell负责的框所框出的类别，有20个类别, 变成[batch_size, 7, 7, 20]，正确的类别对应的位置为1，其它为0

            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])    #由7*7*2 reshape成 1*7*7*2
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])  #在第一个维度上进行复制，变成 [batch_size, 7, 7,2]
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))  #维度为[batch_size, 7, 7, 2]

            #offset_tran如下，只不过batch_size=1
            # [[[[0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]]
            #
            # [[1. 1.]
            #  [1. 1.]
            # [1. 1.]
            # [1. 1.]
            # [1. 1.]
            # [1. 1.]
            # [1. 1.]]
            #
            # [[2. 2.]
            #  [2. 2.]
            # [2. 2.]
            # [2. 2.]
            # [2. 2.]
            # [2. 2.]
            # [2. 2.]]
            #
            # [[3. 3.]
            #  [3. 3.]
            # [3. 3.]
            # [3. 3.]
            # [3. 3.]
            # [3. 3.]
            # [3. 3.]]
            #
            # [[4. 4.]
            #  [4. 4.]
            # [4. 4.]
            # [4. 4.]
            # [4. 4.]
            # [4. 4.]
            # [4. 4.]]
            #
            # [[5. 5.]
            #  [5. 5.]
            # [5. 5.]
            # [5. 5.]
            # [5. 5.]
            # [5. 5.]
            # [5. 5.]]
            #
            # [[6. 6.]
            #  [6. 6.]
            # [6. 6.]
            # [6. 6.]
            # [6. 6.]
            # [6. 6.]
            # [6. 6.]]]]
            #

            predict_boxes_tran = tf.stack(   #相对于整张特征图来说，找到相对于特征图大小的中心点，和宽度以及高度的开方， 其格式为[batch_size, 7, 7, 2, 4]
                [(predict_boxes[..., 0] + offset) / self.cell_size,   #self.cell=7
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),    #宽度的平方，和论文中的开方对应，具体请看论文
                 tf.square(predict_boxes[..., 3])], axis=-1)  #高度的平方，

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)   #计算IOU,  其格式为： [batch_size, 7, 7, 2]

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)  # Computes the maximum of elements across dimensions of a tensor, 在第四个维度上，维度从0开始算
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response   #其维度为[batch_size, 7, 7, 2]  , 如果cell中真实有目标，那么该cell内iou最大的那个框的相应位置为1（就是负责预测该框），其余为0

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask    #其维度为[batch_size, 7 , 7, 2]， 真实没有目标的区域都为1，真实有目标的区域为0

            boxes_tran = tf.stack(     #stack这是一个矩阵拼接的操作， 得到x_center, y_center相对于该cell左上角的偏移值， 宽度和高度是相对于整张图片的比例
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),   #宽度开方，和论文对应
                 tf.sqrt(boxes[..., 3])], axis=-1)  #高度开方，和论文对应

            # class_loss, 计算类别的损失
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(   #平方差损失函数
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale   # self.class_scale为损失函数前面的系数

            # 有目标的时候，置信度损失函数
            object_delta = object_mask * (predict_scales - iou_predict_truth)  #用iou_predict_truth替代真实的置信度，真的妙，佩服的5体投递
            object_loss = tf.reduce_mean(  #平方差损失函数
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # 没有目标的时候，置信度的损失函数
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(       #平方差损失函数
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # 框坐标的损失，只计算有目标的cell中iou最大的那个框的损失，即用这个iou最大的框来负责预测这个框，其它不管，乘以0
            coord_mask = tf.expand_dims(object_mask, 4)  # object_mask其维度为：[batch_size, 7, 7, 2]， 扩展维度之后变成[batch_size, 7, 7, 2, 1]
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)     #predict_boxes维度为： [batch_size, 7, 7, 2, 4]，这些框的坐标都是偏移值
            coord_loss = tf.reduce_mean(  #平方差损失函数
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)  #将各个损失总结起来

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):  #leaky_relu激活函数
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
