def cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """

    status = _libcublas.cublasZgemm_v2(handle,
                                       _CUBLAS_OP[transa],
                                       _CUBLAS_OP[transb], m, n, k, 
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)