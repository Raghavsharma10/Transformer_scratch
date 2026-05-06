def isa(cls, protocol):
    """Does the type 'cls' participate in the 'protocol'?"""
    if not isinstance(cls, type):
        raise TypeError("First argument to isa must be a type. Got %s." %
                        repr(cls))

    if not isinstance(protocol, type):
        raise TypeError(("Second argument to isa must be a type or a Protocol. "
                         "Got an instance of %r.") % type(protocol))
    return issubclass(cls, protocol) or issubclass(AnyType, protocol)