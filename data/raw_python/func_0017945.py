def cudaMemGetInfo():
    """
    Return the amount of free and total device memory.

    Returns
    -------
    free : long
        Free memory in bytes.
    total : long
        Total memory in bytes.

    """

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _libcudart.cudaMemGetInfo(ctypes.byref(free),
                                       ctypes.byref(total))
    cudaCheckStatus(status)
    return free.value, total.value