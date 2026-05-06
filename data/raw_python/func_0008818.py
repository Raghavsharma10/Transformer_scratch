def mask_plane(data, wcs, region, negate=False):
    """
    Mask a 2d image (data) such that pixels within 'region' are set to nan.

    Parameters
    ----------
    data : 2d-array
        Image array.

    wcs : astropy.wcs.WCS
        WCS for the image in question.

    region : :class:`AegeanTools.regions.Region`
        A region within which the image pixels will be masked.

    negate : bool
        If True then pixels *outside* the region are masked.
        Default = False.

    Returns
    -------
    masked : 2d-array
        The original array, but masked as required.
    """
    # create an array but don't set the values (they are random)
    indexes = np.empty((data.shape[0]*data.shape[1], 2), dtype=int)
    # since I know exactly what the index array needs to look like i can construct
    # it faster than list comprehension would allow
    # we do this only once and then recycle it
    idx = np.array([(j, 0) for j in range(data.shape[1])])
    j = data.shape[1]
    for i in range(data.shape[0]):
        idx[:, 1] = i
        indexes[i*j:(i+1)*j] = idx

    # put ALL the pixles into our vectorized functions and minimise our overheads
    ra, dec = wcs.wcs_pix2world(indexes, 1).transpose()
    bigmask = region.sky_within(ra, dec, degin=True)
    if not negate:
        bigmask = np.bitwise_not(bigmask)
    # rework our 1d list into a 2d array
    bigmask = bigmask.reshape(data.shape)
    # and apply the mask
    data[bigmask] = np.nan
    return data