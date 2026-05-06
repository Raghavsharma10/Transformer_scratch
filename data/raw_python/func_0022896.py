def _copy_gl_functions(source, dest, constants=False):
    """ Inject all objects that start with 'gl' from the source
    into the dest. source and dest can be dicts, modules or BaseGLProxy's.
    """
    # Get dicts
    if isinstance(source, BaseGLProxy):
        s = {}
        for key in dir(source):
            s[key] = getattr(source, key)
        source = s
    elif not isinstance(source, dict):
        source = source.__dict__
    if not isinstance(dest, dict):
        dest = dest.__dict__
    # Copy names
    funcnames = [name for name in source.keys() if name.startswith('gl')]
    for name in funcnames:
        dest[name] = source[name]
    # Copy constants
    if constants:
        constnames = [name for name in source.keys() if name.startswith('GL_')]
        for name in constnames:
            dest[name] = source[name]