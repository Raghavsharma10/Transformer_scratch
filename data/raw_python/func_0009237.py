def _double_centered_imp(a, out=None):
    """
    Real implementation of :func:`double_centered`.

    This function is used to make parameter ``out`` keyword-only in
    Python 2.

    """
    out = _float_copy_to_out(out, a)

    dim = np.size(a, 0)

    mu = np.sum(a) / (dim * dim)
    sum_cols = np.sum(a, 0, keepdims=True)
    sum_rows = np.sum(a, 1, keepdims=True)
    mu_cols = sum_cols / dim
    mu_rows = sum_rows / dim

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= mu_rows
    out -= mu_cols
    out += mu

    return out