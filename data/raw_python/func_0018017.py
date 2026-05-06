def cublasZgerc(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on complex general matrix.

    """

    status = _libcublas.cublasZgerc_v2(handle,
                                       m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                               alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)