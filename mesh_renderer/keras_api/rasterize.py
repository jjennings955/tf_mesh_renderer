import keras
from keras import backend as K
import tensorflow as tf
import mesh_renderer

class Rasterize(keras.layers.Layer):
    def __init__(self, res, **kwargs):
        self.resolution = res
        super(Rasterize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rasterize, self).build(input_shape)

    def call(self, x):
        vertices, faces = x[0], x[1]
        normals = x[2]
        colors = x[3]
        eye, center, world_up = x[4], x[5], x[6]
        light_positions, light_intensities = x[7], x[8]
        pixel_positions, pixel_normals, vertex_ids, barycentric_coordinates = mesh_renderer.mesh_renderer_2(vertices,
                                           faces,
                                           normals,
                                           colors,
                                           K.tile(eye[tf.newaxis, :], [vertices.shape[0], 1]),
                                           K.tile(center[tf.newaxis, :], [vertices.shape[0], 1]),
                                           K.tile(world_up[tf.newaxis, :], [vertices.shape[0], 1]),
                                           K.tile(light_positions[tf.newaxis, :, :], [vertices.shape[0], 1, 1]),
                                           K.tile(light_intensities[tf.newaxis, :, :], [vertices.shape[0], 1, 1]),
                                           self.resolution[0],
                                           self.resolution[1])

        return pixel_positions, pixel_normals, vertex_ids, barycentric_coordinates


    def get_output_shape_for(self, input_shape):
         return (input.shape[0], self.resolution[0], self.resolution[1])