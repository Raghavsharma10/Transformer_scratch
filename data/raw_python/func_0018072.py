def cublasCher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasCher2k_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(cuda.cuFloatComplex(alpha.real,                 
                                                                               alpha.imag)),
                                        int(A), lda, int(B), ldb, 
                                        ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                         beta.imag)),
                                        int(C), ldc)
    cublasCheckStatus(status)