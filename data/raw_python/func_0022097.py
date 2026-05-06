def bbox2wktpolygon(bbox):
    """
    Return OGC WKT Polygon of a simple bbox list
    """

    try:
        minx = float(bbox[0])
        miny = float(bbox[1])
        maxx = float(bbox[2])
        maxy = float(bbox[3])

    except:
        LOGGER.debug("Invalid bbox, setting it to a zero POLYGON")
        minx = 0
        miny = 0
        maxx = 0
        maxy = 0

    return 'POLYGON((%.2f %.2f, %.2f %.2f, %.2f %.2f, %.2f %.2f, %.2f %.2f))' \
        % (minx, miny, minx, maxy, maxx, maxy, maxx, miny, minx, miny)