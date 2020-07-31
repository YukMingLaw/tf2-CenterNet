import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models.CenterNet import centernet
import cv2
import numpy as np

#get model
trainmodel,deploymodel,debugmodel = centernet(1)

tf.keras.utils.plot_model(debugmodel)

#load model weight
deploymodel.load_weights('./save/efficient_b2_hswish_fpn.h5',by_name=True, skip_mismatch=True)

debugmodel.save('./save/efficient_b2_hswish_fpn_debug.h5')


#read img and preprocess
cap = cv2.VideoCapture('/home/cvos/Video/124.mp4')
img_size = 512
while True:
    ret,img = cap.read()
    img = cv2.resize(img,(960,540))
    #img = cv2.imread("/home/cvos/PythonProjiects/tf2-yolov3-nano/test_img/Untitled Folder/11.jpg")
    #pre process image
    org_h = img.shape[0]
    org_w = img.shape[1]
    max_side = max(org_h, org_w)
    if org_h > org_w:
        scale = org_w / max_side
        pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
        pts2 = np.array(
            [[img_size * (1 - scale) / 2, 0], [img_size * (1 + scale) / 2, 0],
             [img_size * (1 - scale) / 2, img_size]],
            dtype=np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (img_size, img_size))
    else:
        scale = org_h / max_side
        pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
        offset1 = img_size * (1 - scale) / 2
        offset2 = img_size * (1 + scale) / 2
        pts2 = np.array([[0, offset1], [img_size, offset1], [0, offset2]],
                        dtype=np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (img_size, img_size))
    img = img / 255.0
    input = img[np.newaxis,:]
    #forward
    outputs = deploymodel(input)
    bboxs = outputs[0].numpy()
    #get result
    # hm = outputs[0][0].numpy()
    # hm = cv2.resize(hm,(608,608))
    #demo result
    for bbox in bboxs:
        if bbox[4] >= 0.4:
            x1 = (int)(bbox[0] * 4)
            y1 = (int)(bbox[1] * 4)
            x2 = (int)(bbox[2] * 4)
            y2 = (int)(bbox[3] * 4)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
    # cv2.imshow('hm',hm)
    cv2.imshow('pred',img)
    cv2.waitKey(1)