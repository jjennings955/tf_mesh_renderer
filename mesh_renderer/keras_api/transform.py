import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

class Transform(keras.layers.Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(Transform, self).__init__(**kwargs)

    def build(self, input_shape):
      self.matrix = K.variable(np.random.randn(self.batch_size, 3, 3))
      super(Transform, self).build(input_shape)

    def call(self, x):
        return tf.matmul(x, self.matrix, transpose_b=True)

    def get_output_shape_for(self, input_shape):
        return input_shape