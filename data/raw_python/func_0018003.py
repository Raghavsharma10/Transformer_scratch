def cublasDestroy(handle):
    """
    Release CUBLAS resources.

    Releases hardware resources used by CUBLAS.

    Parameters
    ----------
    handle : void_p
        CUBLAS context.
        
    """

    status = _libcublas.cublasDestroy_v2(ctypes.c_void_p(handle))
    cublasCheckStatus(status)