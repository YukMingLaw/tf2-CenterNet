#tensorflow 1.14
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 这个代码网上说需要加上, 如果模型里有dropout , bn层的话, 我测试过加不加结果都一样, 保险起见还是加上吧
tf.keras.backend.set_learning_phase(0)

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
             return tf.nn.relu(input)
         elif self.activation == 'relu6':
             return tf.nn.relu6(input)
         elif self.activation == 'swish':
             return tf.nn.swish(input)
         elif self.activation == 'hswish':
             return input * tf.nn.relu6(input + 3.) * (1. / 6.)

# 首先是定义你的模型
model = tf.keras.models.load_model('../save/efficient_b2_hswish_fpn_debug.h5',custom_objects={'EfficientNetActivation': EfficientNetActivation()})

def get_flops(model):
     run_meta = tf.RunMetadata()
     opts = tf.profiler.ProfileOptionBuilder.float_operation()

     # We use the Keras session graph in the call to the profiler.
     flops = tf.profiler.profile(graph=tf.keras.backend.get_session().graph,run_meta=run_meta, cmd='op', options=opts)

     return flops.total_float_ops  # Prints the "flops" of the model.

print(get_flops(model))

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
     from tensorflow.python.framework.graph_util import convert_variables_to_constants
     graph = session.graph
     with graph.as_default():
         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
         output_names = output_names or []
         output_names += [v.op.name for v in tf.global_variables()]
         # Graph -> GraphDef ProtoBuf
         input_graph_def = graph.as_graph_def(add_shapes=True)
         if clear_devices:
             for node in input_graph_def.node:
                 node.device = ""
         frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
         return frozen_graph
frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "./","efficient_b2_hswish_fpn_debug.pb", as_text=False)


