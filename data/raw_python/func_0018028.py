def cublasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex symmetric matrix.

    """

    status = _libcublas.cublasCsymv_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)), 
                                        int(A), lda, int(x), incx, 
                                        ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)), 
                                        int(y), incy)
    cublasCheckStatus(status)