import keras
from keras import backend as K
import tensorflow as tf
#
# class FixedCamera(keras.layers.Layer):
#     def __init__(self, eye, center, world_up, **kwargs):
#         assert eye.shape == (1, 3)
#         assert center.shape == (1, 3)
#         assert world_up.shape == (1, 3)
#
#         self._eye = eye
#         self._center = center
#         self._world_up = world_up
#
#         super(FixedCamera, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.eye = tf.tile(tf.constant(self._eye[np.newaxis, :, :], dtype=tf.float32), [input_shape[0], 1, 1])
#         self.center = tf.tile(tf.constant(self._center[np.newaxis, :, :], dtype=tf.float32), [input_shape[0], 1, 1])
#         self.world_up = tf.tile(tf.constant(self._world_up[np.newaxis, :, :], dtype=tf.float32), [input_shape[0], 1, 1])
#
#     def call(self, x):
#         return [self.eye, self.center, self.world_up]
#
#     def get_output_shape_for(self, input):
#         return [self.eye.shape, self.center.shape, self.world_up.shape]


class Camera(object):
    def __init__(self, eye, center, world_up, _type='fixed'):
        # assert eye.shape == (1, 3)
        # assert center.shape == (1, 3)
        # assert world_up.shape == (1, 3)
        # assert _type in ['fixed', 'variable']
        self._type = _type
        if _type == 'fixed':
            self.eye = K.constant(eye, name='camera_eye')
            self.center = K.constant(center, name='camera_center')
            self.world_up = K.constant(world_up, name='camera_up')
        elif _type == 'variable':
            self.eye = K.variable(eye, name='camera_eye')
            self.center = K.variable(center, name='camera_center')
            self.world_up = K.variable(world_up, name='camera_up')
        else:
            raise ValueError("Camera type must be 'fixed' or 'variable'")



    # def build(self, input_shape):
    #     self.eye = tf.tile(tf.constant(self._eye[np.newaxis, :, :], dtype=tf.float32), [input_shape[0], 1, 1])
    #     self.center = tf.tile(tf.constant(self._center[np.newaxis, :, :], dtype=tf.float32), [input_shape[0], 1, 1])
    #     self.world_up = tf.tile(tf.constant(self._world_up[np.newaxis, :, :], dtype=tf.float32), [input_shape[0], 1, 1])
    #
    # def call(self, x):
    #     return [self.eye, self.center, self.world_up]
    #
    # def get_output_shape_for(self, input):
    #     return [self.eye.shape, self.center.shape, self.world_up.shape]