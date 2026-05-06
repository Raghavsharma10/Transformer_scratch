def cublasSsyr(handle, uplo, n, alpha, x, incx, A, lda): 
    """
    Rank-1 operation on real symmetric matrix.

    """
   
    status = _libcublas.cublasSsyr_v2(handle,
                                      _CUBLAS_FILL_MODE[uplo], n, 
                                      ctypes.byref(ctypes.c_float(alpha)),
                                      int(x), incx, int(A), lda)
    cublasCheckStatus(status)