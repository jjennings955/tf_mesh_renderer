import tensorflow as tf

class Transform(keras.layers.Layer):
    def __init__(self, matrix, **kwargs):
        self.matrix = matrix
        super(Transform, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Transform, self).build(input_shape)

    def call(self, x):
        return tf.matmul(x, self.matrix, transpose_b=True)

    def get_output_shape_for(self, input_shape):
        return input_shape
