def _inject():
    """ Copy functions from OpenGL.GL into _pyopengl namespace.
    """
    NS = _pyopengl2.__dict__
    for glname, ourname in _pyopengl2._functions_to_import:
        func = _get_function_from_pyopengl(glname)
        NS[ourname] = func