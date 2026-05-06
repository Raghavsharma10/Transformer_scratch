def cublasZtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """
    Matrix-matrix product for complex triangular matrix.

    """
    
    status = _libcublas.cublasZtrmm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,     
                                                                               alpha.imag)),
                                       int(A), lda, int(B), ldb, int(C), ldc)
    cublasCheckStatus(status)