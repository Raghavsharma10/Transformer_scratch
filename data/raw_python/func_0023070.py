def _set_config(c):
    """Set the OpenGL configuration"""
    glformat = QGLFormat()
    glformat.setRedBufferSize(c['red_size'])
    glformat.setGreenBufferSize(c['green_size'])
    glformat.setBlueBufferSize(c['blue_size'])
    glformat.setAlphaBufferSize(c['alpha_size'])
    if QT5_NEW_API:
        # Qt5 >= 5.4.0 - below options automatically enabled if nonzero.
        glformat.setSwapBehavior(glformat.DoubleBuffer if c['double_buffer']
                                 else glformat.SingleBuffer)
    else:
        # Qt4 and Qt5 < 5.4.0 - buffers must be explicitly requested.
        glformat.setAccum(False)
        glformat.setRgba(True)
        glformat.setDoubleBuffer(True if c['double_buffer'] else False)
        glformat.setDepth(True if c['depth_size'] else False)
        glformat.setStencil(True if c['stencil_size'] else False)
        glformat.setSampleBuffers(True if c['samples'] else False)
    glformat.setDepthBufferSize(c['depth_size'] if c['depth_size'] else 0)
    glformat.setStencilBufferSize(c['stencil_size'] if c['stencil_size']
                                  else 0)
    glformat.setSamples(c['samples'] if c['samples'] else 0)
    glformat.setStereo(c['stereo'])
    return glformat