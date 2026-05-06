def _get_a_code_object_from( thing ) :
    '''
    Given a thing that might be a property, a class method,
    a function or a code object, reduce it to code object.
    If we cannot, return the thing itself.
    '''
    # If we were passed a Method wrapper, get its function
    if isinstance( thing, types.MethodType ) :
        thing = thing.__func__
    # If we were passed a property object, get its getter function
    # (no direct support for the fdel or fset functions)
    if hasattr( thing, 'fget' ) :
        thing = thing.fget
    # If we were passed, or now have, a function, get its code object.
    if isinstance( thing, types.FunctionType ) :
        thing = thing.__code__
    # We should now have a code object, or will never have it.
    return thing