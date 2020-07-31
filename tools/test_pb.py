#tensorflow 1.14
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

sess = tf.Session()
with gfile.FastGFile("efficient_b2_hswish_fpn_debug.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())

input_x = sess.graph.get_tensor_by_name("input_1:0")

out_sigmoid = sess.graph.get_tensor_by_name("conv2d_95/Sigmoid:0")

import cv2
import numpy as np

def trans_img(org_img,img_size):
    org_h = org_img.shape[0]
    org_w = org_img.shape[1]
    max_side = max(org_h, org_w)
    if org_h > org_w:
        scale = org_w / max_side
        pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
        pts2 = np.array(
            [[img_size * (1 - scale) / 2, 0], [img_size * (1 + scale) / 2, 0],
             [img_size * (1 - scale) / 2, img_size]],
            dtype=np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(org_img, M, (img_size, img_size))
    else:
        scale = org_h / max_side
        pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
        offset1 = img_size * (1 - scale) / 2
        offset2 = img_size * (1 + scale) / 2
        pts2 = np.array([[0, offset1], [img_size, offset1], [0, offset2]],
                        dtype=np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(org_img, M, (img_size, img_size))

    return img

def drawHeatMap(hm):
    cv2.normalize(hm, hm, 0, 255, cv2.NORM_MINMAX)
    norm_hm = np.asarray(hm, dtype=np.uint8)
    hm_show = cv2.applyColorMap(norm_hm, cv2.COLORMAP_OCEAN)
    hm_show = cv2.resize(hm_show, (512, 512))
    cv2.imshow('hm', hm_show)

cap = cv2.VideoCapture('/home/cvos/Video/gate/3.mp4')
img_size = 512

import time

while True:
    ret,read_img = cap.read()
    org_img = cv2.resize(read_img,(960,540))
    #pre process image
    image = trans_img(org_img,img_size) / 255.0
    input = image[np.newaxis,:]
    start = time.clock()
    hm = sess.run(out_sigmoid, feed_dict={input_x:input})
    end = time.clock()
    print('Running time:', (end - start)*1000)
    hm = hm[0]#cv2.resize(hm[0], (img_size, img_size))

    drawHeatMap(hm)
    cv2.imshow('pred', org_img)
    cv2.waitKey(1)
videoWriter.release()
