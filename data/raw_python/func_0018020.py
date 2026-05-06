def cublasSspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-packed matrix.

    """

    status = _libcublas.cublasSspmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       ctypes.byref(ctypes.c_float(AP)),
                                       int(x),
                                       incx,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(y),
                                       incy)
    cublasCheckStatus(status)