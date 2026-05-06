def cublasSetStream(handle, id):
    """
    Set current CUBLAS library stream.
    
    Parameters
    ----------
    handle : id
        CUBLAS context.
    id : int
        Stream ID.

    """

    status = _libcublas.cublasSetStream_v2(handle, id)
    cublasCheckStatus(status)