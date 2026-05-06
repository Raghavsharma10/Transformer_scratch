def cublasGetVersion(handle):
    """
    Get CUBLAS version.

    Returns version number of installed CUBLAS libraries.

    Parameters
    ----------
    handle : void_p
        CUBLAS context.

    Returns
    -------
    version : int
        CUBLAS version.

    """
    
    version = ctypes.c_int()
    status = _libcublas.cublasGetVersion_v2(handle, ctypes.byref(version))
    cublasCheckStatus(status)
    return version.value