def cublasDtrmv(handle, uplo, trans, diag, n, A, lda, x, inx):
    """
    Matrix-vector product for real triangular matrix.

    """

    status = _libcublas.cublasDtrmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), inx)
    cublasCheckStatus(status)