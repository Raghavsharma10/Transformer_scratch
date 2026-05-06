def cudaMallocPitch(pitch, rows, cols, elesize):
    """
    Allocate pitched device memory.

    Allocate pitched memory on the device associated with the current active
    context.

    Parameters
    ----------
    pitch : int
        Pitch for allocation.
    rows : int
        Requested pitched allocation height.
    cols : int
        Requested pitched allocation width.
    elesize : int
        Size of memory element.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """

    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMallocPitch(ctypes.byref(ptr),
                                        ctypes.c_size_t(pitch), cols*elesize,
                                        rows)
    cudaCheckStatus(status)
    return ptr, pitch