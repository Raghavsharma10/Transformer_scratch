def cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on real symmetric matrix.

    """
    
    status = _libcublas.cublasDsyrk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(cuda.cuFloatComplex(alpha.real,     
                                                                              alpha.imag)),
                                       int(A), lda, 
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)