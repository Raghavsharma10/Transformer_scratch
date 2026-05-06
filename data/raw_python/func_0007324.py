def get_attribute(obj, name):
    """Retrieve an attribute while bypassing the descriptor protocol.

    As per the built-in |getattr()|_ function, if the input object is a class
    then its base classes might also be searched until the attribute is found.

    Parameters
    ----------
    obj : object
        Object to search the attribute in.
    name : str
        Name of the attribute.

    Returns
    -------
    object
        The attribute found.

    Raises
    ------
    AttributeError
        The attribute couldn't be found.


    .. |getattr()| replace:: ``getattr()``
    .. _getattr(): https://docs.python.org/library/functions.html#getattr
    """
    objs = inspect.getmro(obj) if isinstance(obj, _CLASS_TYPES) else [obj]
    for obj_ in objs:
        try:
            return object.__getattribute__(obj_, name)
        except AttributeError:
            pass

    raise AttributeError("'%s' object has no attribute '%s'"
                         % (type(obj), name))