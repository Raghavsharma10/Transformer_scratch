def cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real general matrix.

    """

    status = _libcublas.cublasDgemv_v2(handle,
                                       _CUBLAS_OP[trans], m, n,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)