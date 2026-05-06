def cublasGetStream(handle):
    """
    Set current CUBLAS library stream.

    Parameters
    ----------
    handle : void_p
        CUBLAS context.
  
    Returns
    -------
    id : int
        Stream ID.
  
    """
    
    id = ctypes.c_int()
    status = _libcublas.cublasGetStream_v2(handle, ctypes.byref(id))
    cublasCheckStatus(status)
    return id.value