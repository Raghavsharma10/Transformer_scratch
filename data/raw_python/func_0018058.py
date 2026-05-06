def cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for symmetric matrix.

    """
    
    status = _libcublas.cublasSsymm_v2(handle,
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       m, n, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb, 
                                       ctypes.byref(ctypes.c_float(beta)), 
                                       int(C), ldc)
    cublasCheckStatus(status)