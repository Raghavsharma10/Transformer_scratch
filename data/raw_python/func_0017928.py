def vector_normalize(mat, max_vec_norm=1.):
    """ Normalize each column vector in mat to length
    max_vec_norm if it is longer than max_vec_norm
    """
    assert mat.flags.c_contiguous
    n, m = mat.shape

    vector_normalize_kernel.prepared_call(
        (m, 1, 1), (32, 1, 1),
        mat.gpudata,
        np.float32(max_vec_norm),
        np.int32(m),
        np.int32(n))