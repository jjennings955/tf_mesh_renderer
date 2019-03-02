import numpy as np
from keras.models import Input, Model
from keras import backend as K
import tensorflow as tf

from lights import Lights
from camera import Camera
from model import Geometry
from warp import Warp

BATCH_SIZE = 10
NUM_WARPS = 5


base_model = Geometry(vertices=[], faces=[])
lights = Lights(positions=[], intensities=[])
camera = Camera(eye=[], center=[], world_up=[])

warp_params = Input(shape=[NUM_WARPS, 1])
warp_module = Warp(num_warps=NUM_WARPS)

warped_model_batch = Warp([base_model.vertices, warp_params])
model = Model(inputs=[warp_params], outputs=[warped_model_batch])


sess = K.get_session()
sess.run(tf.global_variables_initializer())
sess.run([model], feed_dict={warp_params : np.random.randn(BATCH_SIZE, NUM_WARPS, 1)})