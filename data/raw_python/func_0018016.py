def cublasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on complex general matrix.

    """

    status = _libcublas.cublasCgeru_v2(handle,
                                       m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                              alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)