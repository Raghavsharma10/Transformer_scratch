def cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc):
    """
    Matrix-diagonal matrix product for real general matrix.
        
    """

    status = _libcublas.cublasSdgmm(handle,
                                    _CUBLAS_SIDE[mode],
                                    m, n, 
                                    int(A), lda, 
                                    int(x), incx,
                                    int(C), ldc)
    cublasCheckStatus(status)