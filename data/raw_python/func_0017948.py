def cudaPointerGetAttributes(ptr):
    """
    Get memory pointer attributes.

    Returns attributes of the specified pointer.

    Parameters
    ----------
    ptr : ctypes pointer
        Memory pointer to examine.

    Returns
    -------
    memory_type : int
        Memory type; 1 indicates host memory, 2 indicates device
        memory.
    device : int
        Number of device associated with pointer.

    Notes
    -----
    This function only works with CUDA 4.0 and later.

    """

    attributes = cudaPointerAttributes()
    status = \
        _libcudart.cudaPointerGetAttributes(ctypes.byref(attributes), ptr)
    cudaCheckStatus(status)
    return attributes.memoryType, attributes.device