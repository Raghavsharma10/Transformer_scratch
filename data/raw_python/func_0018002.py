def cublasCreate():
    """
    Initialize CUBLAS.

    Initializes CUBLAS and creates a handle to a structure holding
    the CUBLAS library context.

    Returns
    -------
    handle : void_p
        CUBLAS context.
            
    """

    handle = ctypes.c_void_p()
    status = _libcublas.cublasCreate_v2(ctypes.byref(handle))
    cublasCheckStatus(status)
    return handle.value