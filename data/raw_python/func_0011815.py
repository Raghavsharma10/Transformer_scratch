def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)