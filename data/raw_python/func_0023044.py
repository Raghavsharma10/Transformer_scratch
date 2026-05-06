def _check_color_dim(val):
    """Ensure val is Nx(n_col), usually Nx3"""
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError('Value must have second dimension of size 3 or 4')
    return val, val.shape[1]