def IOWR(type, nr, size):
    """
    An ioctl with both read an writes parameters.

    size (ctype type or instance)
        Type/structure of the argument passed to ioctl's "arg" argument.
    """
    return IOC(IOC_READ | IOC_WRITE, type, nr, IOC_TYPECHECK(size))