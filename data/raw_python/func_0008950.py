def find_centroid(region):
    """
    Finds an approximate centroid for a region that is within the region.
    
    Parameters
    ----------
    region : np.ndarray(shape=(m, n), dtype='bool')
        mask of the region.

    Returns
    -------
    i, j : tuple(int, int)
        2d index within the region nearest the center of mass.
    """

    x, y = center_of_mass(region)
    w = np.argwhere(region)
    i, j = w[np.argmin(np.linalg.norm(w - (x, y), axis=1))]
    return i, j