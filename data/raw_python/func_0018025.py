def cublasDspr2(handle, uplo, n, alpha, x, incx, y, incy, AP):
    """
    Rank-2 operation on real symmetric-packed matrix.

    """

    status = _libcublas.cublasDspr2_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_double(alpha)), 
                                       int(x), incx, int(y), incy, int(AP))
    cublasCheckStatus(status)