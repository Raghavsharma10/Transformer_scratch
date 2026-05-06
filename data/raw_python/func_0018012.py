def cublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general matrix.

    """

    status = _libcublas.cublasCgemv_v2(handle,
                                       _CUBLAS_OP[trans], m, n,
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)