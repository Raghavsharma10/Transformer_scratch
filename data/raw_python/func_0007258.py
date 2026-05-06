def consolidate_subdivide_geometry(geometry, max_query_area_size):
    """
    Consolidate a geometry into a convex hull, then subdivide it into
    smaller sub-polygons if its area exceeds max size (in geometry's units).

    Parameters
    ----------
    geometry : shapely Polygon or MultiPolygon
        the geometry to consolidate and subdivide
    max_query_area_size : float
        max area for any part of the geometry, in the units the geometry is
        in: any polygon bigger will get divided up for multiple queries to
        the Overpass API (default is 50,000 * 50,000 units
        (ie, 50km x 50km in area, if units are meters))

    Returns
    -------
    geometry : Polygon or MultiPolygon
    """

    # let the linear length of the quadrats (with which to subdivide the
    # geometry) be the square root of max area size
    quadrat_width = math.sqrt(max_query_area_size)

    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError('Geometry must be a shapely Polygon or MultiPolygon')

    # if geometry is a MultiPolygon OR a single Polygon whose area exceeds
    # the max size, get the convex hull around the geometry
    if isinstance(
            geometry, MultiPolygon) or \
            (isinstance(
                geometry, Polygon) and geometry.area > max_query_area_size):
        geometry = geometry.convex_hull

    # if geometry area exceeds max size, subdivide it into smaller sub-polygons
    if geometry.area > max_query_area_size:
        geometry = quadrat_cut_geometry(geometry, quadrat_width=quadrat_width)

    if isinstance(geometry, Polygon):
        geometry = MultiPolygon([geometry])

    return geometry