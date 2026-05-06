def _get_function_from_pyopengl(funcname):
    """ Try getting the given function from PyOpenGL, return
    a dummy function (that shows a warning when called) if it
    could not be found.
    """
    func = None
    
    # Get function from GL
    try:
        func = getattr(_GL, funcname)
    except AttributeError:
        # Get function from FBO
        try:
            func = getattr(_FBO, funcname)
        except AttributeError:
            func = None
    
    # Try using "alias"
    if not bool(func):
        # Some functions are known by a slightly different name
        # e.g. glDepthRangef, glClearDepthf
        if funcname.endswith('f'):
            try:
                func = getattr(_GL, funcname[:-1])
            except AttributeError:
                pass

    # Set dummy function if we could not find it
    if func is None:
        func = _make_unavailable_func(funcname)
        logger.warning('warning: %s not available' % funcname)
    return func