import tensorflow as tf
import numpy as np
import math
import cv2
from utils.utils import compute_gt

class CenterNetGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            train_list=[],
            num_classes=1,
            multi_scale=False,
            multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
            batch_size=1,
            shuffle = False,
            misc_effect = None,
            visual_effect = None,
            input_size = 512,
            max_num_box = 100,
            debug = False
    ):
        self.current_index = 0
        self.train_list = train_list
        self.multi_scale = multi_scale
        self.multi_image_sizes = multi_image_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.misc_effect = misc_effect
        self.visual_effect = visual_effect
        self.input_size = input_size
        self.num_classes = num_classes
        self.max_num_box = max_num_box
        self.debug = debug
        if(len(train_list) == 0):
            print('error train set is empty!')
            exit()
        if(len(train_list) < batch_size):
            print('train set count {0} is less than batch size {1}'.format(len(train_list),batch_size))
            exit()

    def __len__(self):
        return math.ceil(len(self.train_list) / self.batch_size)

    def __getitem__(self, index):
        if self.multi_scale:
            self.input_size = np.random.choice(self.multi_image_sizes)
        batch_img,batch_box = self.load_batch(index,self.input_size)

        gt = compute_gt(batch_box,self.input_size // 4,self.num_classes,self.max_num_box)
        return [batch_img,*gt],np.zeros(self.batch_size)


    def load_batch(self,index,input_size):

        if self.shuffle:
            train_batch = []
            for i in range(self.batch_size):
                random_index = np.random.randint(0,len(self.train_list))
                train_batch.append(self.train_list[random_index])
        else:
            train_batch = self.train_list[index * self.batch_size : index * self.batch_size + self.batch_size]

        return self.get_batch_img_and_label(train_batch,input_size)

    def get_batch_img_and_label(self,batch_list,img_size,max_num_box = 20):
        #load images
        batch_img = self.load_batch_image(batch_list)
        batch_label = self.load_batch_labels(batch_list)

        if self.visual_effect is not None:
            batch_img, batch_label = self.batch_visual_effect(batch_img,batch_label)

        if self.misc_effect is not None:
            batch_img, batch_label = self.batch_misc_effect(batch_img, batch_label)

        batch_img, batch_label = self.batch_preprocess(batch_img, batch_label)

        batch_img = np.array(batch_img)
        batch_label = np.array(batch_label)
        return batch_img,batch_label

    def load_batch_image(self, batch_list):
        return [self.load_image(image_path) for image_path in batch_list]

    def load_batch_labels(self, batch_list):
        return [self.load_label(label_path) for label_path in batch_list]

    def load_image(self,image_path):
        img = cv2.imread(image_path)
        return img

    def load_label(self,label_path):
        _path = label_path.split('.')
        with open(_path[0] + '.txt') as f:
            _line = f.readline()
            boxes = []
            while _line:
                _line_split = _line.split()
                obj_class = int(_line_split[0])
                _box = [float(i) for i in _line_split[1:]]
                _box.append(obj_class)
                boxes.append(_box)
                _line = f.readline()
            box_data = np.zeros((self.max_num_box, 5))
            if len(boxes) > 0:
                if len(boxes) > self.max_num_box:
                    boxes = boxes[:self.max_num_box]
                box_data[:len(boxes)] = np.array(boxes)
        return box_data

    def batch_visual_effect(self,batch_image,batch_label):
        for index in range(len(batch_image)):
            batch_image[index] = self.visual_effect(batch_image[index])
        return batch_image,batch_label

    def batch_misc_effect(self,batch_image,batch_label):
        for index in range(len(batch_image)):
            batch_image[index], batch_label[index][:,:4] = self.misc_effect(batch_image[index],batch_label[index][:,:4])
        return batch_image, batch_label

    def batch_preprocess(self,batch_image,batch_label):
        for index in range(len(batch_image)):
            batch_image[index], batch_label[index][:,:4] = self.preprocess(batch_image[index],batch_label[index][:,:4])
        return batch_image, batch_label

    def preprocess(self,image,label):
        org_h = image.shape[0]
        org_w = image.shape[1]
        max_side = max(org_h, org_w)
        if org_h > org_w:
            scale = org_w / max_side
            pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
            offset1 = self.input_size * (1 - scale) / 2
            offset2 = self.input_size * (1 + scale) / 2
            pts2 = np.array([[offset1, 0], [offset2, 0], [offset1, self.input_size]],
                            dtype=np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, M, (self.input_size,self.input_size))
        else:
            scale = org_h / max_side
            pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
            offset1 = self.input_size * (1 - scale) / 2
            offset2 = self.input_size * (1 + scale) / 2
            pts2 = np.array([[0, offset1], [self.input_size, offset1], [0, offset2]],
                            dtype=np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, M, (self.input_size, self.input_size))

        image = image / 255.0

        for i in range(len(label)):
            if org_h > org_w:
                label[i][0] = (label[i][0] * self.input_size * scale + offset1) / self.input_size
                label[i][2] = label[i][2] * scale
            else:
                label[i][1] = (label[i][1] * self.input_size * scale + offset1) / self.input_size
                label[i][3] = label[i][3] * scale

        return image,label
