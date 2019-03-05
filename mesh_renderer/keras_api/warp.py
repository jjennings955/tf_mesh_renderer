import keras
from keras import backend as K
import tensorflow as tf

def warp_rbf(a, x):
    v = a
    keypoint, warp_vector, params = x[0], x[1], x[2]
    d = tf.norm((v - keypoint), axis=2, keepdims=True)
    k_d = tf.exp(-1.5 * d ** 2)
    nudge = tf.reshape(params, [-1, 1, 1]) * k_d * x[1]
    return v + nudge


class Warp(keras.layers.Layer):
    def __init__(self, keypoints, warp_vectors, **kwargs):
        self.keypoints = keypoints
        self.warp_vectors = warp_vectors
        super(Warp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Warp, self).build(input_shape)

    def call(self, x):
        vertices, warp_params = x[0], x[1]
        vertices_repeated = tf.tile(vertices[tf.newaxis, :, :], [tf.shape(x[1])[0], 1, 1])
        warped_vertices = tf.foldl(warp_rbf,
                                   elems=[self.keypoints, self.warp_vectors,
                                          K.permute_dimensions(warp_params, (1, 0, 2))],
                                   initializer=vertices_repeated)
        self.warp_strengths = 0.5 - tf.exp(-2.5 * tf.norm(warped_vertices - vertices, axis=2, keepdims=True) ** 2)

        return warped_vertices

    def get_output_shape_for(self, input_shape):
        return [input_shape[1][0]] + input_shape[0]