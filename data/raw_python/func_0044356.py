def catalog_to_cells(catalog, radius, order, include_fallback=True, **kwargs):
    """
    Convert a catalog to a set of cells.

    This function is intended to be used via `catalog_to_moc` but
    is available for separate usage.  It takes the same arguments
    as that function.

    This function uses the Healpy `query_disc` function to get a list
    of cells for each item in the catalog in turn.  Additional keyword
    arguments, if specified, are passed to `query_disc`.  This can include,
    for example, `inclusive` (set to `True` to include cells overlapping
    the radius as well as those with centers within it) and `fact`
    (to control sampling when `inclusive` is specified).

    If cells at the given order are bigger than the given radius, then
    `query_disc` may find none inside the radius.  In this case,
    if `include_fallback` is `True` (the default), the cell at each
    position is included.

    If the given radius is zero (or smaller) then Healpy `query_disc`
    is not used -- instead the fallback position is used automatically.
    """

    nside = 2 ** order

    # Ensure catalog is in ICRS coordinates.
    catalog = catalog.icrs

    # Ensure radius is in radians.
    if isinstance(radius, Quantity):
        radius = radius.to(radian).value
    else:
        radius = radius * pi / (180.0 * 3600.0)

    # Convert coordinates to position vectors.
    phi = catalog.ra.radian
    theta = (pi / 2) - catalog.dec.radian

    vectors = ang2vec(theta, phi)

    # Ensure we can iterate over vectors (it might be a single position).
    if catalog.isscalar:
        vectors = [vectors]

    # Query for a list of cells for each catalog position.
    cells = set()
    for vector in vectors:
        if radius > 0.0:
            # Try "disc" query.
            vector_cells = query_disc(nside, vector, radius, nest=True, **kwargs)

            if vector_cells.size > 0:
                cells.update(vector_cells.tolist())
                continue

            elif not include_fallback:
                continue

        # The query didn't find anything -- include the cell at the
        # given position at least.
        cell = vec2pix(nside, vector[0], vector[1], vector[2], nest=True)
        cells.add(cell.item())

    return cells