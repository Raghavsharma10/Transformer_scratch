def get_object(path="", obj=None):
    """Return an object from a dot path.

    Path can either be a full path, in which case the `get_object` function
    will try to import the modules in the path and follow it to the final
    object. Or it can be a path relative to the object passed in as the second
    argument.

    Args:
        path (str): Full or relative dot path to the desired object
        obj (object): Starting object. Dot path is calculated relatively to
            this object.

    Returns:
        Object at the end of the path, or list of non hidden objects if we use
        the star query.

    Example for full paths::

        >>> get_object('os.path.join')
        <function join at 0x1002d9ed8>
        >>> get_object('tea.process')
        <module 'tea.process' from 'tea/process/__init__.pyc'>

    Example for relative paths when an object is passed in::

        >>> import os
        >>> get_object('path.join', os)
        <function join at 0x1002d9ed8>

    Example for a star query. (Star query can be used only as the last element
    of the path::

        >>> get_object('tea.dsa.*')
        []
        >>> get_object('tea.dsa.singleton.*')
        [<class 'tea.dsa.singleton.Singleton'>,
         <class 'tea.dsa.singleton.SingletonMetaclass'>
         <module 'six' from '...'>]
        >>> get_object('tea.dsa.*')
        [<module 'tea.dsa.singleton' from '...'>]    # Since we imported it
    """
    if not path:
        return obj
    path = path.split(".")
    if obj is None:
        obj = importlib.import_module(path[0])
        path = path[1:]
    for item in path:
        if item == "*":
            # This is the star query, returns non hidden objects
            return [
                getattr(obj, name)
                for name in dir(obj)
                if not name.startswith("__")
            ]
        if isinstance(obj, types.ModuleType):
            submodule = "{}.{}".format(_package(obj), item)
            try:
                obj = importlib.import_module(submodule)
            except Exception as import_error:
                try:
                    obj = getattr(obj, item)
                except Exception:
                    # FIXME: I know I should probably merge the errors, but
                    #        it's easier just to throw the import error since
                    #        it's most probably the one user wants to see.
                    #        Create a new LoadingError and throw a combination
                    #        of the import error and attribute error.
                    raise import_error
        else:
            obj = getattr(obj, item)
    return obj