def cublasSsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on real symmetric matrix.

    """
    
    status = _libcublas.cublasSsyr2k_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(ctypes.c_float(alpha)),
                                        int(A), lda, int(B), ldb, 
                                        ctypes.byref(ctypes.c_float(beta)), 
                                        int(C), ldc)
    cublasCheckStatus(status)