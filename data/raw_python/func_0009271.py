def _cdist_naive(x, y, exponent=1):
    """Pairwise distance, custom implementation."""
    squared_norms = ((x[_np.newaxis, :, :] - y[:, _np.newaxis, :]) ** 2).sum(2)

    exponent = exponent / 2
    try:
        exponent = squared_norms.take(0).from_float(exponent)
    except AttributeError:
        pass

    return squared_norms ** exponent