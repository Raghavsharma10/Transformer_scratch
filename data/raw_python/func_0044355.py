def _catalog_to_cells_neighbor(catalog, radius, order):
    """
    Convert a catalog to a list of cells.

    This is the original implementation of the `catalog_to_cells`
    function which does not make use of the Healpy `query_disc` routine.

    Note: this function uses a simple flood-filling approach and is
    very slow, especially when used with a large radius for catalog objects
    or a high resolution order.
    """

    if not isinstance(radius, Quantity):
        radius = radius * arcsecond

    nside = 2 ** order

    # Ensure catalog is in ICRS coordinates.
    catalog = catalog.icrs

    # Determine central cell for each catalog entry.
    phi = catalog.ra.radian
    theta = (pi / 2) - catalog.dec.radian

    cells = np.unique(ang2pix(nside, theta, phi, nest=True))

    # Iteratively consider the neighbors of cells within our
    # catalog regions.
    new_cells = cells
    rejected = np.array((), dtype=np.int64)
    while True:
        # Find new valid neighboring cells which we didn't already
        # consider.
        neighbors = np.unique(np.ravel(
            get_all_neighbours(nside, new_cells, nest=True)))

        neighbors = np.extract(
            [(x != -1) and (x not in cells) and (x not in rejected)
             for x in neighbors], neighbors)

        # Get the coordinates of each of these neighbors and compare them
        # to the catalog entries.
        (theta, phi) = pix2ang(nside, neighbors, nest=True)

        coords = SkyCoord(phi, (pi / 2) - theta, frame='icrs', unit='rad')

        (idx, sep2d, dist3d) = coords.match_to_catalog_sky(catalog)

        within_range = (sep2d < radius)

        # If we didn't find any new cells within range,
        # end the iterative process.
        if not np.any(within_range):
            break

        new_cells = neighbors[within_range]
        cells = np.concatenate((cells, new_cells))
        rejected = np.concatenate((
            rejected, neighbors[np.logical_not(within_range)]))

    return cells