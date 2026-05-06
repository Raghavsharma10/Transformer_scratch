def cublasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular matrix.

    """
    
    status = _libcublas.cublasCtrmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)