def cublasZher(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasZher_v2(handle, 
                                      _CUBLAS_FILL_MODE[uplo], 
                                      n, alpha, int(x), incx, int(A), lda)
    cublasCheckStatus(status)