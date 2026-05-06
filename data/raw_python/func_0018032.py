def cublasCsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on complex symmetric matrix.

    """

    status = _libcublas.cublasCsyr_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                        int(x), incx, int(A), lda)
    cublasCheckStatus(status)