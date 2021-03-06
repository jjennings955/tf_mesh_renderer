import numpy as np
from keras.models import Input, Model
from keras import backend as K
import tensorflow as tf

from lights import Lights
from camera import Camera
from model import Geometry
from warp import Warp
from render import Render

BATCH_SIZE = 10
NUM_WARPS = 5
num_vertices = 100
num_faces = 150
num_lights = 1


base_model = Geometry(vertices=np.random.randn(num_vertices, 3), faces=np.random.randint(0, num_vertices, size=[num_faces, 3]))
lights = Lights(positions=np.random.randn(num_lights, 3), intensities=np.random.randn(num_lights, 3))
camera = Camera(eye=np.random.randn(1, 3), center=np.random.randn(1, 3), world_up=np.random.randn(1, 3))
trans = Transform(batch_size=BATCH_SIZE)

warp_params = Input(shape=[NUM_WARPS, 1])
warped_vertices = Warp(num_warps=NUM_WARPS)([K.identity(base_model.vertices), warp_params])
world_coords = trans(warped_vertices)
colors = K.constant(np.random.randn(BATCH_SIZE, num_vertices, 3))
rendered = Render(512, 512)([world_coords, base_model.faces],
                            base_model.calculate_normals(world_coords),
                            colors,
                            [camera.eye, camera.center, camera.world_up],
                            [lights.positions, lights.intensities])

#model = Model(inputs=[warp_params], outputs=[renderer])

sess = K.get_session()
sess.run(tf.global_variables_initializer())
sess.run([rendered], feed_dict={warp_params : np.random.randn(BATCH_SIZE, NUM_WARPS, 1)})

