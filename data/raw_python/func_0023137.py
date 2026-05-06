def eglInitialize(display):
    """ Initialize EGL and return EGL version tuple.
    """
    majorVersion = (_c_int*1)()
    minorVersion = (_c_int*1)()
    res = _lib.eglInitialize(display, majorVersion, minorVersion)
    if res == EGL_FALSE:
        raise RuntimeError('Could not initialize')
    return majorVersion[0], minorVersion[0]