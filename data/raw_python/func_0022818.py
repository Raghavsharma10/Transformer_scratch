def _set_config(c):
    """Set gl configuration"""
    gl_attribs = [glcanvas.WX_GL_RGBA,
                  glcanvas.WX_GL_DEPTH_SIZE, c['depth_size'],
                  glcanvas.WX_GL_STENCIL_SIZE, c['stencil_size'],
                  glcanvas.WX_GL_MIN_RED, c['red_size'],
                  glcanvas.WX_GL_MIN_GREEN, c['green_size'],
                  glcanvas.WX_GL_MIN_BLUE, c['blue_size'],
                  glcanvas.WX_GL_MIN_ALPHA, c['alpha_size']]
    gl_attribs += [glcanvas.WX_GL_DOUBLEBUFFER] if c['double_buffer'] else []
    gl_attribs += [glcanvas.WX_GL_STEREO] if c['stereo'] else []
    return gl_attribs