def yield_typed(obj_or_cls):
    """
    Generator that yields typed object names of the class (or object's class).

    Args:
        obj_or_cls (object): Class object or instance of class

    Returns:
        name (array): Names of class attributes that are strongly typed
    """
    if not isinstance(obj_or_cls, type):
        obj_or_cls = type(obj_or_cls)
    for attrname in dir(obj_or_cls):
        if hasattr(obj_or_cls, attrname):
            attr = getattr(obj_or_cls, attrname)
            # !!! Important hardcoded value here !!!
            if (isinstance(attr, property) and isinstance(attr.__doc__, six.string_types)
                and "__typed__" in attr.__doc__):
                yield attrname