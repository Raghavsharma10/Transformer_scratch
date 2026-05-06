def cublasSgbmv(handle, trans, m, n, kl, ku, alpha, A, lda,
                x, incx, beta, y, incy):
    """
    Matrix-vector product for real general banded matrix.

    """

    status = _libcublas.cublasSgbmv_v2(handle,
                                       trans, m, n, kl, ku,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda,
                                       int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)