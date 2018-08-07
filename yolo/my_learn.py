#这个是供作者学习测速python语法用的

import numpy as np
import tensorflow as tf

# a=np.reshape(np.array(
#             [np.arange(7)] * 7 * 2),
#             (2, 7, 7))
#
# print(np.arange(7))
# print('____-')
# print(a)
# print("transpose")
# print(np.transpose(a))
# print('--------------')
# print([0,1]*7*3)
#
# print("___________")

# x = tf.constant([1, 4])
# y = tf.constant([2, 5])
# z = tf.constant([3, 6])
# a=tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
# b=tf.stack([x, y, z], axis=-1)  # [[1, 2, 3], [4, 5, 6]]
# sess=tf.Session()
# print(sess.run(b))

# label = np.zeros((2, 2, 3))
# print(label)
# a=label[:, ::-1, :]
# print(a)

# offset = np.transpose(np.reshape(np.array(   #reshape之后再转置，变成7*7*2的三维数组
#             [np.arange(7)] * 7 * 2),
#             (2, 7, 7)), (1, 2, 0))
#
# offset = tf.reshape(
#                 tf.constant(offset, dtype=tf.float32),
#                 [1, 7, 7, 2])    #由7*7*2 reshape成 1*7*7*2
#
# # offset_tran = tf.transpose(offset, (0, 2, 1, 3))
#
# sess=tf.Session()
# print(sess.run(offset))

# a=[[[1,2],[2,3]],[[3,4],[5,6]]]
# b=[[2],[3]]
#
#
# c=tf.reshape(a,[2,2,2])
# d=tf.reshape(b,[2,1])
#
# f=tf.expand_dims(c, 3)
#
# sess=tf.Session()
#
# g=tf.reduce_sum(c,axis=[1,2])
#
# print(sess.run(f))

# probs=np.ones([2,3])
#
# filter_mat_probs = np.array(probs >= 0, dtype='bool')
# filter_mat_boxes = np.nonzero(filter_mat_probs)
# print(filter_mat_boxes)

a = np.array([[[0,1],[1,0]],[[0,1],[1,0]],[[0,0],[1,0]]])
b = np.nonzero(a)
print(np.array(b).ndim)
print(b)










