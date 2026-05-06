def cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on real general matrix.

    """
    
    status = _libcublas.cublasSger_v2(handle,
                                      m, n,
                                      ctypes.byref(ctypes.c_float(alpha)),
                                      int(x), incx,
                                      int(y), incy, int(A), lda)
    cublasCheckStatus(status)