def cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on real symmetric matrix.

    """
    
    status = _libcublas.cublasSsyrk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, 
                                       ctypes.byref(ctypes.c_float(beta)), 
                                       int(C), ldc)
    cublasCheckStatus(status)