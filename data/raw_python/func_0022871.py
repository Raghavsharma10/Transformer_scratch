def _to_args(x):
    """Convert to args representation"""
    if not isinstance(x, (list, tuple, np.ndarray)):
        x = [x]
    return x