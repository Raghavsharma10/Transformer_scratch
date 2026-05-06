def cublasSsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on real symmetric matrix.

    """

    status = _libcublas.cublasSsyr2_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(x), incx, int(y), incy,
                                       int(A), lda)
    cublasCheckStatus(status)