# -*- coding: utf-8 -*-

import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):  # Yolon_net and pascal_voc_data
        self.net = net  # 训练的网络
        self.data = data  # train或者test的数据
        self.weights_file = cfg.WEIGHTS_FILE  # 权重文件
        self.max_iter = cfg.MAX_ITER  # 迭代次数,迭代次数可自定义
        self.initial_learning_rate = cfg.LEARNING_RATE  # 学习率，0.0001
        self.decay_steps = cfg.DECAY_STEPS  # 衰变步数
        self.decay_rate = cfg.DECAY_RATE  # 衰变率
        self.staircase = cfg.STAIRCASE  # true
        self.summary_iter = cfg.SUMMARY_ITER  # SUMMARY_ITER, default 10
        self.save_iter = cfg.SAVE_ITER  # save itger, default 1000
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR,
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))  # add time， data/pascal_voc/output/date_time
        if not os.path.exists(self.output_dir):  # 不存在则创建目录
            os.makedirs(self.output_dir)
        self.save_cfg()  # 保存配置

        self.variable_to_restore = tf.global_variables()  # 初始化tensorflow的全局变量
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)  # 定义tf.saver
        self.ckpt_file = os.path.join(self.output_dir, 'yolo.ckpt')  # 定义保存模型输出的权重文件
        self.summary_op = tf.summary.merge_all()  # 将tensorflow各个操作联合起来，省事
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)  # 将内容写入到文件中，每60秒更新一次

        self.global_step = tf.train.create_global_step()  # 创建全局的步骤
        self.learning_rate = tf.train.exponential_decay(  # 设定变化的学习率，这个可以yolo论文中的相关指导来设定
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(  # 采用的优化方法是随机梯度下降
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(  # 将tensorflow的operation联合起来
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())  # 初始化全局变量

        if self.weights_file is not None:  # 权重文件不等于None的时候
            print('Restoring weights from: ' + self.weights_file)  # 加载预训练模型
            self.saver.restore(self.sess, self.weights_file)  # 从预训练模型中restore

        self.writer.add_graph(self.sess.graph)  # 加图

    def train(self):  # start training

        train_timer = Timer()  # train_timer
        load_timer = Timer()  # load_timer

        for step in range(1, self.max_iter + 1):  # 开始训练
            print("step: ", step)
            load_timer.tic()
            images, labels = self.data.get()  # 获取到batch_size大小的图片和对应的label
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}  # 喂数据

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:  # 将一些训练信息打印出来

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = '''{} Epoch: {}, Step: {}, Learning rate: {},'''
                    ''' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'''
                    '''' Load: {:.3f}s/iter, Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:  # 只是训练，不打印出信息
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:  # 保留检查点，以供测试时用
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(  # 保存会话，将模型文件保存
                    self.sess, self.ckpt_file, global_step=self.global_step)
                print("save done!!!")

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:  # 把配置信息写入到文件中
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):  # 更新配置文件路径

    print("应该是加载了YOLO_small.ckpt")
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():  # 自定义参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)  # 定义权重文件
    parser.add_argument('--data_dir', default="data", type=str)  # 定义数据文件夹
    parser.add_argument('--threshold', default=0.2, type=float)  # 阈值
    parser.add_argument('--iou_threshold', default=0.5, type=float)  # IOU阈值
    parser.add_argument('--gpu', default='', type=str)  # 是否用gpu训练
    args = parser.parse_args()

    if args.gpu is not None:  # 是否用gpu训练
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()  # Yolo网络
    pascal = pascal_voc('train')  # 获得训练的数据， 包含了经过水平翻转后的训练实例

    solver = Solver(yolo, pascal)  # 准备训练的环境，包括设置优化器，学习率等内容

    print('Start training ...')
    solver.train()  # start training
    print('done!!!')

    # f = open('result.txt', 'w')
    # f.write('train finished!!!!')
    # f.close()


if __name__ == '__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()  # main 函数
