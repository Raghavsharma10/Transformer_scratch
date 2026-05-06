def cudaDriverGetVersion():
    """
    Get installed CUDA driver version.

    Return the version of the installed CUDA driver as an integer. If
    no driver is detected, 0 is returned.

    Returns
    -------
    version : int
        Driver version.

    """

    version = ctypes.c_int()
    status = _libcudart.cudaDriverGetVersion(ctypes.byref(version))
    cudaCheckStatus(status)
    return version.value