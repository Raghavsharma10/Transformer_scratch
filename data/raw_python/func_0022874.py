def get_gl_configuration():
    """Read the current gl configuration

    This function uses constants that are not in the OpenGL ES 2.1
    namespace, so only use this on desktop systems.

    Returns
    -------
    config : dict
        The currently active OpenGL configuration.
    """
    # XXX eventually maybe we can ask `gl` whether or not we can access these
    gl.check_error('pre-config check')
    config = dict()
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    fb_param = gl.glGetFramebufferAttachmentParameter
    # copied since they aren't in ES:
    GL_FRONT_LEFT = 1024
    GL_DEPTH = 6145
    GL_STENCIL = 6146
    GL_SRGB = 35904
    GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING = 33296
    GL_STEREO = 3123
    GL_DOUBLEBUFFER = 3122
    sizes = dict(red=(GL_FRONT_LEFT, 33298),
                 green=(GL_FRONT_LEFT, 33299),
                 blue=(GL_FRONT_LEFT, 33300),
                 alpha=(GL_FRONT_LEFT, 33301),
                 depth=(GL_DEPTH, 33302),
                 stencil=(GL_STENCIL, 33303))
    for key, val in sizes.items():
        config[key + '_size'] = fb_param(gl.GL_FRAMEBUFFER, val[0], val[1])
    val = fb_param(gl.GL_FRAMEBUFFER, GL_FRONT_LEFT,
                   GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING)
    if val not in (gl.GL_LINEAR, GL_SRGB):
        raise RuntimeError('unknown value for SRGB: %s' % val)
    config['srgb'] = True if val == GL_SRGB else False  # GL_LINEAR
    config['stereo'] = True if gl.glGetParameter(GL_STEREO) else False
    config['double_buffer'] = (True if gl.glGetParameter(GL_DOUBLEBUFFER)
                               else False)
    config['samples'] = gl.glGetParameter(gl.GL_SAMPLES)
    gl.check_error('post-config check')
    return config