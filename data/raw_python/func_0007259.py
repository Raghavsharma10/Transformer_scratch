def quadrat_cut_geometry(geometry, quadrat_width, min_num=3,
                         buffer_amount=1e-9):
    """
    Split a Polygon or MultiPolygon up into sub-polygons of a specified size,
    using quadrats.

    Parameters
    ----------
    geometry : shapely Polygon or MultiPolygon
        the geometry to split up into smaller sub-polygons
    quadrat_width : float
        the linear width of the quadrats with which to cut up the geometry
        (in the units the geometry is in)
    min_num : float
        the minimum number of linear quadrat lines (e.g., min_num=3 would
        produce a quadrat grid of 4 squares)
    buffer_amount : float
        buffer the quadrat grid lines by quadrat_width times buffer_amount

    Returns
    -------
    multipoly : shapely MultiPolygon
    """

    # create n evenly spaced points between the min and max x and y bounds
    lng_max, lat_min, lng_min, lat_max = geometry.bounds
    x_num = math.ceil((lng_min-lng_max) / quadrat_width) + 1
    y_num = math.ceil((lat_max-lat_min) / quadrat_width) + 1
    x_points = np.linspace(lng_max, lng_min, num=max(x_num, min_num))
    y_points = np.linspace(lat_min, lat_max, num=max(y_num, min_num))

    # create a quadrat grid of lines at each of the evenly spaced points
    vertical_lines = [LineString([(x, y_points[0]), (x, y_points[-1])])
                      for x in x_points]
    horizont_lines = [LineString([(x_points[0], y), (x_points[-1], y)])
                      for y in y_points]
    lines = vertical_lines + horizont_lines

    # buffer each line to distance of the quadrat width divided by 1 billion,
    # take their union, then cut geometry into pieces by these quadrats
    buffer_size = quadrat_width * buffer_amount
    lines_buffered = [line.buffer(buffer_size) for line in lines]
    quadrats = unary_union(lines_buffered)
    multipoly = geometry.difference(quadrats)

    return multipoly