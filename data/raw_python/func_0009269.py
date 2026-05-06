def _transform_to_2d(t):
    """Convert vectors to column matrices, to always have a 2d shape."""
    t = np.asarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t