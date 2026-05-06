def cublasDtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Solve real triangular-packed system with one right-hand side.

    """

    status = _libcublas.cublasDtpsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)