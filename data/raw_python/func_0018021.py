def cublasDspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-packed matrix.

    """

    status = _libcublas.cublasDspmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       ctypes.byref(ctypes.c_double(AP)),
                                       int(x),
                                       incx,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(y),
                                       incy)
    cublasCheckStatus(status)