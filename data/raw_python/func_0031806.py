def isopoly_to_gdf(collec_poly, levels, field_name="levels"):
    """
    Convert a collection of matplotlib.contour.QuadContourSet to a GeoDataFrame
    Set an attribute `field_name` on each feature, according to `levels` values
    (`levels` must have the same number of features as the collection of contours)

    Parameters
    ----------
    collection_polygons : matplotlib.contour.QuadContourSet
        The result of a grid interpolation from matplotlib.
    levels : array-like
        The value to use as attributes for the constructed GeoDataFrame.
    field_name : str
        The name of the field to be fill by values contained in
        `levels` variable (default: "levels").

    Returns
    -------
    gdf_contours : GeoDataFrame
        The result as a GeoDataFrame.
    """
    polygons, data = [], []

    for i, polygon in enumerate(collec_poly.collections):
        mpoly = []
        for path in polygon.get_paths():
            path.should_simplify = False
            poly = path.to_polygons()
            exterior, holes = [], []
            if len(poly) > 0 and len(poly[0]) > 3:
                exterior = poly[0]
                if len(poly) > 1:
                    holes = [h for h in poly[1:] if len(h) > 3]
            mpoly.append(Polygon(exterior, holes))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            data.append(levels[i])
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            data.append(levels[i])

    return GeoDataFrame(geometry=polygons,
                        data=data,
                        columns=[field_name])