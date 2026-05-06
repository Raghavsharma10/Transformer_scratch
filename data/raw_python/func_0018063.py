def cublasZsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on complex symmetric matrix.

    """
    
    status = _libcublas.cublasZsyrk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(cuda.cuDoubleComplex(alpha.real,    
                                                                               alpha.imag)),
                                       int(A), lda,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)