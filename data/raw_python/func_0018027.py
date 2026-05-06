def cublasDsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric matrix.
    
    """

    status = _libcublas.cublasDsymv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_double(alpha)), 
                                       int(A), lda, int(x), incx, 
                                       ctypes.byref(ctypes.c_double(beta)), 
                                       int(y), incy)
    cublasCheckStatus(status)