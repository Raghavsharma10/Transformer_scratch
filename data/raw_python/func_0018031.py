def cublasDsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on real symmetric matrix.

    """

    status = _libcublas.cublasDsyr_v2(handle,
                                      _CUBLAS_FILL_MODE[uplo], n, 
                                      ctypes.byref(ctypes.c_double(alpha)), 
                                      int(x), incx, int(A), lda)
    cublasCheckStatus(status)