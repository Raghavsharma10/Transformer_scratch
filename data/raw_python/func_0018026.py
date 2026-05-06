def cublasSsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric matrix.
    
    """
    
    status = _libcublas.cublasSsymv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)), 
                                       int(y), incy)
    cublasCheckStatus(status)