import tensorflow as tf
from tensorflow.keras.layers import Conv2D,DepthwiseConv2D,BatchNormalization,ReLU,Add,GlobalAveragePooling2D,Reshape,Multiply
from tensorflow.keras.regularizers import l2
import math

class EfficientNetActivation(tf.keras.layers.Layer):
    def __init__(self,activation='swish',**kwargs):
        super(EfficientNetActivation, self).__init__(**kwargs)
        self.activation = activation

    def get_config(self):
        config = {"activation": self.activation}
        base_config = super(EfficientNetActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):
        if self.activation == 'relu':
            return ReLU()(input)
        elif self.activation == 'relu6':
            return ReLU(max_value=6)(input)
        elif self.activation == 'swish':
            return tf.nn.swish(input)
        elif self.activation == 'hswish':
            return input * tf.nn.relu6(input + 3.) * (1. / 6.)


def round_filters(filters,width_coe,divisor=8,skip=False):
  """Round number of filters based on depth multiplier."""
  orig_f = filters
  multiplier = width_coe
  if skip or not multiplier:
    return filters
  filters *= multiplier
  new_filters = int(filters + divisor / 2) // divisor * divisor
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)

def round_repeats(repeats, depth_coe, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = depth_coe
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))

def MBBlock(input, kernel_size, num_repeat, stride, output_filters, expand_ratio, skip_connect=True, Activation='relu', se_ratio=0.25):
    assert Activation in ['relu', 'relu6', 'swish', 'hswish']
    x = input
    for r in range(num_repeat):
        input_filters = x.shape[3]
        skip_x = x
        x = Conv2D(filters=(input_filters * expand_ratio),kernel_size=1,use_bias=False,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = EfficientNetActivation(activation='hswish')(x)
        if r != 0:
            stride = 1
        x = DepthwiseConv2D(kernel_size=kernel_size, strides=(stride, stride), use_bias=False,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = EfficientNetActivation(activation='hswish')(x)
        if 0 < se_ratio <= 1:
            num_reduced_filters = max(1, input_filters * se_ratio)
            _x = GlobalAveragePooling2D()(x)
            _x = Reshape((1,1,_x.shape[1]))(_x)
            _x = Conv2D(filters=int(num_reduced_filters),kernel_size=1,use_bias=True,padding='same',kernel_initializer='he_normal',activation='relu')(_x)
            _x = Conv2D(filters=int(input_filters * expand_ratio),kernel_size=1,use_bias=True,padding='same',kernel_initializer='he_normal',activation='sigmoid')(_x)
            x = Multiply()([x,_x])
        x = Conv2D(filters=output_filters, kernel_size=1, use_bias=False,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        if stride == 1 and skip_connect and r != 0:
            x = Add()([x,skip_x])
    return x


def EfficientNet(model_name,input):
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    width_coe,depth_coe,input_size,dropout_rate = params_dict[model_name]

    x = Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', use_bias=False,kernel_initializer='he_normal',kernel_regularizer=l2(5e-4))(input)
    x = BatchNormalization()(x)
    x = EfficientNetActivation(activation='hswish')(x)
    output = MBBlock(x,
                     kernel_size=3,
                     num_repeat=round_repeats(1, depth_coe),
                     output_filters=round_filters(16, width_coe=width_coe),
                     stride=1,
                     expand_ratio=1)
    output = MBBlock(output,
                     kernel_size=3,
                     num_repeat=round_repeats(2, depth_coe),
                     output_filters=round_filters(24, width_coe=width_coe),
                     stride=2,
                     expand_ratio=6)
    output = MBBlock(output,
                     kernel_size=5,
                     num_repeat=round_repeats(2, depth_coe),
                     output_filters=round_filters(40, width_coe=width_coe),
                     stride=2,
                     expand_ratio=6)
    output = MBBlock(output,
                     kernel_size=3,
                     num_repeat=round_repeats(3, depth_coe),
                     output_filters=round_filters(80, width_coe=width_coe),
                     stride=2,
                     expand_ratio=6)
    output = MBBlock(output,
                     kernel_size=5,
                     num_repeat=round_repeats(3, depth_coe),
                     output_filters=round_filters(112, width_coe=width_coe),
                     stride=1,
                     expand_ratio=6)
    output = MBBlock(output,
                     kernel_size=5,
                     num_repeat=round_repeats(4, depth_coe),
                     output_filters=round_filters(192, width_coe=width_coe),
                     stride=2,
                     expand_ratio=6)
    output = MBBlock(output,
                     kernel_size=3,
                     num_repeat=round_repeats(1, depth_coe),
                     output_filters=round_filters(320, width_coe=width_coe),
                     stride=1,
                     expand_ratio=6)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model




