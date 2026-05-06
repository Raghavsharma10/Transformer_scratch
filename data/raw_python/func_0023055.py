def _set_config(c):
    """Set gl configuration for SDL2"""
    func = sdl2.SDL_GL_SetAttribute
    func(sdl2.SDL_GL_RED_SIZE, c['red_size'])
    func(sdl2.SDL_GL_GREEN_SIZE, c['green_size'])
    func(sdl2.SDL_GL_BLUE_SIZE, c['blue_size'])
    func(sdl2.SDL_GL_ALPHA_SIZE, c['alpha_size'])
    func(sdl2.SDL_GL_DEPTH_SIZE, c['depth_size'])
    func(sdl2.SDL_GL_STENCIL_SIZE, c['stencil_size'])
    func(sdl2.SDL_GL_DOUBLEBUFFER, 1 if c['double_buffer'] else 0)
    samps = c['samples']
    func(sdl2.SDL_GL_MULTISAMPLEBUFFERS, 1 if samps > 0 else 0)
    func(sdl2.SDL_GL_MULTISAMPLESAMPLES, samps if samps > 0 else 0)
    func(sdl2.SDL_GL_STEREO, c['stereo'])