def cudaGetDevice():
    """
    Get current CUDA device.

    Return the identifying number of the device currently used to
    process CUDA operations.

    Returns
    -------
    dev : int
        Device number.

    """

    dev = ctypes.c_int()
    status = _libcudart.cudaGetDevice(ctypes.byref(dev))
    cudaCheckStatus(status)
    return dev.value