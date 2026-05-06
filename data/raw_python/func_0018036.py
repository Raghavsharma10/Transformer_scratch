def cublasStbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for real triangular-banded matrix.

    """
    
    status = _libcublas.cublasStbmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)