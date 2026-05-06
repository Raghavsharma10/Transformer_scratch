def gpuarray_ptr(g):
    """
    Return ctypes pointer to data in GPUAarray object.

    """

    addr = int(g.gpudata)
    if g.dtype == np.int8:
        return ctypes.cast(addr, POINTER(ctypes.c_byte))
    if g.dtype == np.uint8:
        return ctypes.cast(addr, POINTER(ctypes.c_ubyte))
    if g.dtype == np.int16:
        return ctypes.cast(addr, POINTER(ctypes.c_short))
    if g.dtype == np.uint16:
        return ctypes.cast(addr, POINTER(ctypes.c_ushort))
    if g.dtype == np.int32:
        return ctypes.cast(addr, POINTER(ctypes.c_int))
    if g.dtype == np.uint32:
        return ctypes.cast(addr, POINTER(ctypes.c_uint))
    if g.dtype == np.int64:
        return ctypes.cast(addr, POINTER(ctypes.c_long))
    if g.dtype == np.uint64:
        return ctypes.cast(addr, POINTER(ctypes.c_ulong))
    if g.dtype == np.float32:
        return ctypes.cast(addr, POINTER(ctypes.c_float))
    elif g.dtype == np.float64:
        return ctypes.cast(addr, POINTER(ctypes.c_double))
    elif g.dtype == np.complex64:
        return ctypes.cast(addr, POINTER(cuFloatComplex))
    elif g.dtype == np.complex128:
        return ctypes.cast(addr, POINTER(cuDoubleComplex))
    else:
        raise ValueError('unrecognized type')