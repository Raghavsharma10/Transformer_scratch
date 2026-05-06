def _set_config(config):
    """Set gl configuration"""
    pyglet_config = pyglet.gl.Config()

    pyglet_config.red_size = config['red_size']
    pyglet_config.green_size = config['green_size']
    pyglet_config.blue_size = config['blue_size']
    pyglet_config.alpha_size = config['alpha_size']

    pyglet_config.accum_red_size = 0
    pyglet_config.accum_green_size = 0
    pyglet_config.accum_blue_size = 0
    pyglet_config.accum_alpha_size = 0

    pyglet_config.depth_size = config['depth_size']
    pyglet_config.stencil_size = config['stencil_size']
    pyglet_config.double_buffer = config['double_buffer']
    pyglet_config.stereo = config['stereo']
    pyglet_config.samples = config['samples']
    return pyglet_config