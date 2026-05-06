def cublasZtrsm(handle, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve complex triangular system with multiple right-hand sides.

    """
    
    status = _libcublas.cublasZtrsm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,                    
                                                                               alpha.imag)),
                                       int(A), lda, int(B), ldb)
    cublasCheckStatus(status)