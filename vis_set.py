import cv2
import numpy as np
from generator.generator import CenterNetGenerator
from utils.visual_effect_preprocess import VisualEffect
from utils.misc_effect_preprocess import MiscEffect

with open('/home/cvos/Datasets/coco_car/train.txt') as f:
    _line = f.readlines()
train_set = [i.rstrip('\n') for i in _line]
train_generator = CenterNetGenerator(train_list = train_set,
                                shuffle=False,
                                num_classes = 1,
                                batch_size=1,
                                multi_scale=True,
                                visual_effect = VisualEffect(),
                                misc_effect = MiscEffect(border_value=0))
for i in range(len(train_set)):
    b_img_gt,target = train_generator.__getitem__(i)
    img = b_img_gt[0][0]
    hm = b_img_gt[1][0]
    cv2.imshow('vis',img)
    cv2.imshow('hm',hm)
    print(hm)
    cv2.waitKey(1000)