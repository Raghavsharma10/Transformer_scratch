def decompose_delta(deltak):
    '''Decomposes the k-th order trend filtering matrix into a c-compatible set
    of arrays.'''
    if not isspmatrix_coo(deltak):
        deltak = coo_matrix(deltak)
    dk_rows = deltak.shape[0]
    dk_rowbreaks = np.cumsum(deltak.getnnz(1), dtype="int32")
    dk_cols = deltak.col.astype('int32')
    dk_vals = deltak.data.astype('double')
    return dk_rows, dk_rowbreaks, dk_cols, dk_vals