def cublasDsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on real symmetric matrix.

    """

    status = _libcublas.cublasDsyr2k_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(ctypes.c_double(alpha)),
                                        int(A), lda, int(B), ldb, 
                                        ctypes.byref(ctypes.c_double(beta)), 
                                        int(C), ldc)
    cublasCheckStatus(status)