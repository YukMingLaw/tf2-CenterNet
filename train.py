import tensorflow as tf
import numpy as np
from models.CenterNet import centernet
from generator.generator import CenterNetGenerator
from utils.visual_effect_preprocess import VisualEffect
from utils.misc_effect_preprocess import MiscEffect
import os

train_path = '/home/cvos/Datasets/coco_car/train.txt'
num_classes = 1
batch_size = 6
epochs = 120

def m_scheduler(epoch):
    #warm up
    if epoch == 0:
        return 0.0001
    elif epoch <= 10:
        return 0.0001 * epoch
    elif 10 < epoch <= 40:
        return 0.001
    elif 40 < epoch <= 70:
        return 0.0001
    else:
        return 0.00001

def create_callbacks():
    callbacks = []
    #add tensorboard callback
    tensorboard_callback = None
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        update_freq='batch'
    )
    callbacks.append(tensorboard_callback)
    # save the model
    # ensure directory created first; otherwise h5py will error after epoch.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            './save',
            'efficient_b2_hswish_fpn.h5'
        ),
        verbose=1,
    )
    callbacks.append(checkpoint)
    learnrate = tf.keras.callbacks.LearningRateScheduler(m_scheduler)
    callbacks.append(learnrate)
    return callbacks

def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #get train set
    with open(train_path) as f:
        _line = f.readlines()
    train_set = [i.rstrip('\n') for i in _line]
    train_generator = CenterNetGenerator(train_list=train_set,
                                    num_classes=num_classes,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    multi_scale=True,
                                    visual_effect=VisualEffect(),
                                    misc_effect=MiscEffect(border_value=0),
                                    input_size=512,
                                    max_num_box=100)

    #creat model
    train_model,deploy_model,debug_model = centernet(num_classes = num_classes,max_objects=100)

    #if you want to resume the train,open the code
    #train_model.load_weights('./save/efficient_fpn.h5')

    train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),loss={'centernet_loss':lambda y_true,y_pred:y_pred})

    callbacks = create_callbacks()
    
    # start training
    train_model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    return 0

if __name__ == '__main__':
    main()
