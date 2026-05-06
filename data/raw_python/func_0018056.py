def cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real general matrix.

    """

    status = _libcublas.cublasDgemm_v2(handle,
                                       _CUBLAS_OP[transa],
                                       _CUBLAS_OP[transb], m, n, k, 
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(C), ldc)
    cublasCheckStatus(status)