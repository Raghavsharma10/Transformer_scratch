def use_gl(target='gl2'):
    """ Let Vispy use the target OpenGL ES 2.0 implementation
    
    Also see ``vispy.use()``.
    
    Parameters
    ----------
    target : str
        The target GL backend to use.

    Available backends:
    * gl2 - Use ES 2.0 subset of desktop (i.e. normal) OpenGL
    * gl+ - Use the desktop ES 2.0 subset plus all non-deprecated GL
      functions on your system (requires PyOpenGL)
    * es2 - Use the ES2 library (Angle/DirectX on Windows)
    * pyopengl2 - Use ES 2.0 subset of pyopengl (for fallback and testing)
    * dummy - Prevent usage of gloo.gl (for when rendering occurs elsewhere)
    
    You can use vispy's config option "gl_debug" to check for errors
    on each API call. Or, one can specify it as the target, e.g. "gl2
    debug". (Debug does not apply to 'gl+', since PyOpenGL has its own
    debug mechanism)
    """
    target = target or 'gl2'
    target = target.replace('+', 'plus')
    
    # Get options
    target, _, options = target.partition(' ')
    debug = config['gl_debug'] or 'debug' in options
    
    # Select modules to import names from
    try:
        mod = __import__(target, globals(), level=1)
    except ImportError as err:
        msg = 'Could not import gl target "%s":\n%s' % (target, str(err))
        raise RuntimeError(msg)

    # Apply
    global current_backend
    current_backend = mod
    _clear_namespace()
    if 'plus' in target:
        # Copy PyOpenGL funcs, extra funcs, constants, no debug
        _copy_gl_functions(mod._pyopengl2, globals())
        _copy_gl_functions(mod, globals(), True)
    elif debug:
        _copy_gl_functions(_debug_proxy, globals())
    else:
        _copy_gl_functions(mod, globals())