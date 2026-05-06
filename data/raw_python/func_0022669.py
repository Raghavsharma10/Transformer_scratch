def _inject():
    """ Inject functions and constants from PyOpenGL but leave out the
    names that are deprecated or that we provide in our API.
    """
    
    # Get namespaces
    NS = globals()
    GLNS = _GL.__dict__
    
    # Get names that we use in our API
    used_names = []
    used_names.extend([names[0] for names in _pyopengl2._functions_to_import])
    used_names.extend([name for name in _pyopengl2._used_functions])
    NS['_used_names'] = used_names
    #
    used_constants = set(_constants.__dict__)
    # Count
    injected_constants = 0
    injected_functions = 0
    
    for name in dir(_GL):
        
        if name.startswith('GL_'):
            # todo: find list of deprecated constants
            if name not in used_constants:
                NS[name] = GLNS[name]
                injected_constants += 1
        
        elif name.startswith('gl'):
            # Functions
            if (name + ',') in _deprecated_functions:
                pass  # Function is deprecated
            elif name in used_names:
                pass  # Function is in our GL ES 2.0 API
            else:
                NS[name] = GLNS[name]
                injected_functions += 1