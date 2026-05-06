def _vector(x, type='row'):
    """Convert an object to a row or column vector."""
    if isinstance(x, (list, tuple)):
        x = np.array(x, dtype=np.float32)
    elif not isinstance(x, np.ndarray):
        x = np.array([x], dtype=np.float32)
    assert x.ndim == 1
    if type == 'column':
        x = x[:, None]
    return x