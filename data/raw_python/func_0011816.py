def to_24bit_gray(mat: np.ndarray):
    """returns a matrix that contains RGB channels, and colors scaled
    from 0 to 255"""
    return np.repeat(np.expand_dims(_normalize(mat), axis=2), 3, axis=2)