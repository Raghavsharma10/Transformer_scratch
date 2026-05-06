def get_undecorated_calling_module():
    """Returns the module name of the caller's calling module.
    If a.py makes a call to b() in b.py, b() can get the name of the
    calling module (i.e. a) by calling get_undecorated_calling_module().

    The module also includes its full path.

    As the name suggests, this does not work for decorated functions.
    """
    frame = inspect.stack()[2]
    module = inspect.getmodule(frame[0])
    # Return the module's file and its path
    # and omit the extension...
    # so /a/c.py becomes /a/c
    return module.__file__.rsplit('.', 1)[0]