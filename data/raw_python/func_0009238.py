def _u_centered_imp(a, out=None):
    """
    Real implementation of :func:`u_centered`.

    This function is used to make parameter ``out`` keyword-only in
    Python 2.

    """
    out = _float_copy_to_out(out, a)

    dim = np.size(a, 0)

    u_mu = np.sum(a) / ((dim - 1) * (dim - 2))
    sum_cols = np.sum(a, 0, keepdims=True)
    sum_rows = np.sum(a, 1, keepdims=True)
    u_mu_cols = np.ones((dim, 1)).dot(sum_cols / (dim - 2))
    u_mu_rows = (sum_rows / (dim - 2)).dot(np.ones((1, dim)))

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= u_mu_rows
    out -= u_mu_cols
    out += u_mu

    # The diagonal is zero
    out[np.eye(dim, dtype=bool)] = 0

    return out