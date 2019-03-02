from lights import Lights
from camera import Camera
NUM_WARPS = 5


base_model = BaseModel(vertices=[], faces=[], colors=[])
lights = Lights(positions=[], intensities=[])
camera = Camera(eye=[], center=[], world_up=[])

warp_params = Input(shape=[NUM_WARPS, 1])
warp_module = Warp(num_warps=NUM_WARPS)([base_model.vertices, warp_params])

warped_model_batch =