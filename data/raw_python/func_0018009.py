def cublasZgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, 
                x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general banded matrix.

    """

    status = _libcublas.cublasZgbmv_v2(handle,
                                       trans, m, n, kl, ku,
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                              int(y), incy)
    cublasCheckStatus(status)