def eglGetDisplay(display=EGL_DEFAULT_DISPLAY):
    """ Connect to the EGL display server.
    """
    res = _lib.eglGetDisplay(display)
    if not res or res == EGL_NO_DISPLAY:
        raise RuntimeError('Could not create display')
    return res