def _rescale_layout(pos, scale=1):
    """
    Normalize the given coordinate list to the range [0, `scale`].

    Parameters
    ----------
    pos : array
        Coordinate list
    scale : number
        The upperbound value for the coordinates range

    Returns
    -------
    pos : array
        The rescaled (normalized) coordinates in the range [0, `scale`].

    Notes
    -----
    Changes `pos` in place.
    """

    pos -= pos.min(axis=0)
    pos *= scale / pos.max()

    return pos