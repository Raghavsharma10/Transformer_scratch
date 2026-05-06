def IOW(type, nr, size):
    """
    An ioctl with write parameters.

    size (ctype type or instance)
        Type/structure of the argument passed to ioctl's "arg" argument.
    """
    return IOC(IOC_WRITE, type, nr, IOC_TYPECHECK(size))