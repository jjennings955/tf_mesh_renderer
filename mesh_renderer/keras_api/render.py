import mesh_renderer

class Render(keras.layers.Layer):
    def __init__(self, res, **kwargs):
        self.resolution = res
        super(Render, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Render, self).build(input_shape)

    def call(self, x):
        vertices, faces = x[0]
        normals = x[1]
        colors = x[2]
        eye, center, world_up = x[3]
        light_positions, light_intensities = x[4]

        return mesh_renderer.mesh_renderer(vertices,
                                           faces,
                                           normals,
                                           colors,
                                           K.tile(eye[tf.newaxis, :, :], [vertices.shape[0], 1, 1]),
                                           K.tile(center[tf.newaxis, :, :], [vertices.shape[0], 1, 1]),
                                           K.tile(world_up[tf.newaxis, :, :], [vertices.shape[0], 1, 1]),
                                           K.tile(light_positions[tf.newaxis, :, :], [vertices.shape[0], 1, 1]),
                                           K.tile(light_intensities[tf.newaxis, :, :], [vertices.shape[0], 1, 1]),
                                           self.resolution[0],
                                           self.resolution[1])

    def get_output_shape_for(self, input_shape):
        return (input.shape[0], self.resolution[0], self.resolution[1])

Renderer()([world_coords, base_model.faces], base_model.calculate_normals(woorld_coords), colors, [camera.eye, camera.center, camera.world_up], [lights.positions, lights.intensities])