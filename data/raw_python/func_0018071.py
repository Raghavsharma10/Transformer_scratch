def cublasZherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasZherk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans],
                                       n, k, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, 
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(C), ldc)
    cublasCheckStatus(status)