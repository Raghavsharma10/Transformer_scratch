def cublasChpr(handle, uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on Hermitian-packed matrix.
    
    """
    
    status = _libcublas.cublasChpr_v2(handle, 
                                      _CUBLAS_FILL_MODE[uplo], 
                                      n, ctypes.byref(ctypes.c_float(alpha)),
                                      int(x), incx, int(AP))
    cublasCheckStatus(status)