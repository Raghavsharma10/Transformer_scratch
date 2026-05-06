def scale(s, dtype=None):
    """Non-uniform scaling along the x, y, and z axes

    Parameters
    ----------
    s : array-like, shape (3,)
        Scaling in x, y, z.
    dtype : dtype | None
        Output type (if None, don't cast).

    Returns
    -------
    M : ndarray
        Transformation matrix describing the scaling.
    """
    assert len(s) == 3
    return np.array(np.diag(np.concatenate([s, (1.,)])), dtype)