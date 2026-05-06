def cublasZtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Solve complex triangular-packed system with one right-hand size.

    """
    
    status = _libcublas.cublasZtpsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)