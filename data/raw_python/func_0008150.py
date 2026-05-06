def _ensure_sparse_format(spmatrix, accept_sparse, dtype, order, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
        matrix input will raise an error.  If the input is sparse but not in
        the allowed format, it will be converted to the first listed format.

    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if accept_sparse is None:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    sparse_type = spmatrix.format
    if dtype is None:
        dtype = spmatrix.dtype
    if sparse_type in accept_sparse:
        # correct type
        if dtype == spmatrix.dtype:
            # correct dtype
            if copy:
                spmatrix = spmatrix.copy()
        else:
            # convert dtype
            spmatrix = spmatrix.astype(dtype)
    else:
        # create new
        spmatrix = spmatrix.asformat(accept_sparse[0]).astype(dtype)
    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data)
    if hasattr(spmatrix, "data"):
        spmatrix.data = np.array(spmatrix.data, copy=False, order=order)
    return spmatrix