def bbox2wktpolygon(bbox):
    """
    Return OGC WKT Polygon of a simple bbox list of strings
    """

    minx = float(bbox[0])
    miny = float(bbox[1])
    maxx = float(bbox[2])
    maxy = float(bbox[3])
    return 'POLYGON((%.2f %.2f, %.2f %.2f, %.2f %.2f, %.2f %.2f, %.2f %.2f))' \
        % (minx, miny, minx, maxy, maxx, maxy, maxx, miny, minx, miny)