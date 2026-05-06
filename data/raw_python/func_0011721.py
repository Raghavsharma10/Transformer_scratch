def _rescale(ar):
    """Shift and rescale array ar to the interval [-1, 1]"""
    max = np.nanmax(ar)
    min = np.nanmin(ar)
    midpoint = (max + min) / 2.0
    return 2.0 * (ar - midpoint) / (max - min)