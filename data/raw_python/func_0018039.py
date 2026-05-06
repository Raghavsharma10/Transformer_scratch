def cublasDtpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for real triangular-packed matrix.

    """

    status = _libcublas.cublasDtpmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)