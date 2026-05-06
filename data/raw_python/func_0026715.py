def IOR(type, nr, size):
    """
    An ioctl with read parameters.

    size (ctype type or instance)
        Type/structure of the argument passed to ioctl's "arg" argument.
    """
    return IOC(IOC_READ, type, nr, IOC_TYPECHECK(size))