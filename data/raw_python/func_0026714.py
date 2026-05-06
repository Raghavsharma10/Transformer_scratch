def IOC_TYPECHECK(t):
    """
    Returns the size of given type, and check its suitability for use in an
    ioctl command number.
    """
    result = ctypes.sizeof(t)
    assert result <= _IOC_SIZEMASK, result
    return result