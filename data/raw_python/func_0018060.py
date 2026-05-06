def cublasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex symmetric matrix.

    """
    
    status = _libcublas.cublasCsymm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,                   
                                                                              alpha.imag)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real, 
                                                                        beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)