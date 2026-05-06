def glBufferData(target, data, usage):
    """ Data can be numpy array or the size of data to allocate.
    """
    if isinstance(data, int):
        size = data
        data = ctypes.c_voidp(0)
    else:
        if not data.flags['C_CONTIGUOUS'] or not data.flags['ALIGNED']:
            data = data.copy('C')
        data_ = data
        size = data_.nbytes
        data = data_.ctypes.data
    res = _lib.glBufferData(target, size, data, usage)