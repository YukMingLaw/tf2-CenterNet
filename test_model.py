from models.resnet import centernet
import cv2
import numpy as np
from generators.utils import get_affine_transform, affine_transform
#get model
trainmodel,deploymodel,debugmodel = centernet(1)
#load model weight
deploymodel.load_weights('checkpoints/2020-04-11/save_model.h5',by_name=True, skip_mismatch=True)

#print model struct
debugmodel.summary()
#read img and preprocess
img = cv2.imread("1.jpg")
#pre process image
img_w = img.shape[1]
img_h = img.shape[0]
image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
s = max(image.shape[0], image.shape[1]) * 1.0
trans_input = get_affine_transform(c, s, (512, 512))
image = cv2.warpAffine(image, trans_input, (512, 512), flags=cv2.INTER_LINEAR)
image = image.astype(np.float32)
input = image / 255.0
input = input[np.newaxis,:]
#forward
outputs = deploymodel(input)
#get result
bboxs = outputs[0].numpy()
#demo result
offset = (s-img_h)/2
for bbox in bboxs:
    if bbox[4]>=0.1:
        x1 = (int)(bbox[0] * 4 / 512 * s)
        y1 = (int)((bbox[1] * 4 / 512 * s)-offset)
        x2 = (int)(bbox[2] * 4 / 512 * s)
        y2 = (int)((bbox[3] * 4 / 512 * s)-offset)
        print(x1,y1,x2,y2)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('pred',img)
cv2.waitKey(0)