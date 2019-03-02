#vertices_world_space, faces, normals_world_space,vertex_diffuse_colors,
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

def broadcast_matmul(A, B):
    "Compute A @ B, broadcasting over the first `N-2` ranks"
    with tf.variable_scope("broadcast_matmul"):
        return tf.reduce_sum(A[..., tf.newaxis] * B[..., tf.newaxis, :, :],
                             axis=-2)
class Geometry(object):
    def __init__(self, vertices, faces, **kwargs):

        self.vertices = K.variable(vertices, name='vertices')
        self.faces = K.constant(faces, name='faces', dtype=tf.int32)
        self.adj_list = self.calculate_adj_list(vertices, faces)

    def calculate_adj_list(self, vertices, faces):
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]
        adj_list = np.zeros((num_vertices, num_faces), dtype=np.float32)

        for i in range(num_faces):
            for j in range(3):
                idx = faces[i][j]
                adj_list[idx, i] = 1
        adj_list /= np.sum(adj_list, axis=1, keepdims=True)
        self.adj_list = tf.constant(adj_list[np.newaxis, :, :])

    def calculate_normals_on_batch(self, vertices_batch):
        tv = tf.gather(vertices_batch, self.faces, axis=1)
        v_1 = tv[:, :, 1] - tv[:, :, 0]
        v_2 = tv[:, :, 2] - tv[:, :, 0]
        face_normals = tf.cross(v_1, v_2)
        vertex_normals = broadcast_matmul(self.adj_list, face_normals)
        normals = tf.nn.l2_normalize(vertex_normals, dim=1)
        return normals
