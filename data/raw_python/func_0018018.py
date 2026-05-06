def cublasSsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-banded matrix.

    """

    status = _libcublas.cublasSsbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], n, k,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)