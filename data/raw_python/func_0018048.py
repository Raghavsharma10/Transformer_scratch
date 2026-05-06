def cublasZhpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian-packed matrix.

    """
    
    status = _libcublas.cublasZhpmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                            alpha.imag)),
                                       int(AP), int(x), incx,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real, 
                                                                         beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)