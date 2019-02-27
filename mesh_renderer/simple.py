import math
import os

import numpy as np
import tensorflow as tf

import mesh_renderer
import camera_utils
import test_utils

tf.reset_default_graph()
# Set up a basic cube centered at the origin, with vertex normals pointing
# outwards along the line from the origin to the cube vertices:
cube_vertices = tf.constant(
    [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
     [1, -1, -1], [1, 1, -1], [1, 1, 1]],
    dtype=tf.float32)
cube_normals = tf.nn.l2_normalize(cube_vertices, dim=1)
cube_triangles = tf.constant(
    [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
     [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
    dtype=tf.int32)
model_transforms = camera_utils.euler_matrices(
    [[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]])[:, :3, :3]

vertices_world_space = tf.matmul(
    tf.stack([cube_vertices, cube_vertices]),
    model_transforms,
    transpose_b=True)

normals_world_space = tf.matmul(
    tf.stack([cube_normals, cube_normals]),
    model_transforms,
    transpose_b=True)

# camera position:
eye = tf.constant(2 * [[0.0, 0.0, 6.0]], dtype=tf.float32)
center = tf.constant(2 * [[0.0, 0.0, 0.0]], dtype=tf.float32)
world_up = tf.constant(2 * [[0.0, 1.0, 0.0]], dtype=tf.float32)
image_width = 1920
image_height = 1080
light_positions = tf.constant([[[0.0, 0.0, 6.0]], [[0.0, 0.0, 6.0]]])
light_intensities = tf.ones([2, 1, 3], dtype=tf.float32)
vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)

rendered = mesh_renderer.mesh_renderer(
    vertices_world_space, cube_triangles, normals_world_space,
    vertex_diffuse_colors, eye, center, world_up, light_positions,
    light_intensities, image_width, image_height)
import skimage, skimage.io
with tf.Session() as sess:
    images = sess.run(rendered, feed_dict={})
    for image_id in range(images.shape[0]):
        target_image_name = 'Gray_Cube_%i.png' % image_id
        baseline_image_path = os.path.join('out',
                                           target_image_name)
        skimage.io.imsave(images[id])
