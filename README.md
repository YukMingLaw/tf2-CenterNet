# TF2-CenterNet
#### 引用：
论文地址: [CenterNet](https://arxiv.org/abs/1904.07850)

官方实现:[xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). 

本个仓库修改自:[xuannianz/keras-CenterNet](https://github.com/xuannianz/keras-CenterNet).
#### 声明：
1.删减了一部分其他数据集用于生成训练数据代码，只保留了生成COCO数据集相关的代码;

2.移除了keras_resnet依赖，目前使用`tf.keras.tf.keras.applications`底下的backbone模型;
#### 依赖：
1.tensorflow-gpu=2.1.0

2.opencv-python>=3.4.0

## Train
### 1.build dataset (COCO)
如果你使用的是yolo的image-labels.txt这种形式的数据集，可通过tools文件夹底下的转换代码转换成coco数据集格式；
### 2.train
运行 `python train.py coco yourcocopath` 即可开始训练

恢复训练 `python train.py --resume your_weight_path coco yourcocopath` 即可恢复训练
## Test
修改test_model.py中的cv2.imread('xx.jpg'),
运行 `python test_model.py`
即可测试网络推理效果
## License
This project is released under the Apache License. Some parts are borrowed from [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). Please take their licenses into consideration too when use this project.
