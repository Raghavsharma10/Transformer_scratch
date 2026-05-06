def implements(obj, protocol):
    """Does the object 'obj' implement the 'prococol'?"""
    if isinstance(obj, type):
        raise TypeError("First argument to implements must be an instance. "
                        "Got %r." % obj)
    return isinstance(obj, protocol) or issubclass(AnyType, protocol)