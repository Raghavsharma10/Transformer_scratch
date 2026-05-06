def create_geom_filter(request, mapped_class, geom_attr):
    """Create MapFish geometry filter based on the request params. Either
    a box or within or geometry filter, depending on the request params.
    Additional named arguments are passed to the spatial filter.

    Arguments:

    request
        the request.

    mapped_class
        the SQLAlchemy mapped class.

    geom_attr
        the key of the geometry property as defined in the SQLAlchemy
        mapper. If you use ``declarative_base`` this is the name of
        the geometry attribute as defined in the mapped class.
    """
    tolerance = float(request.params.get('tolerance', 0.0))
    epsg = None
    if 'epsg' in request.params:
        epsg = int(request.params['epsg'])
    box = request.params.get('bbox')
    shape = None
    if box is not None:
        box = [float(x) for x in box.split(',')]
        shape = Polygon(((box[0], box[1]), (box[0], box[3]),
                         (box[2], box[3]), (box[2], box[1]),
                         (box[0], box[1])))
    elif 'lon' in request.params and 'lat' in request.params:
        shape = Point(float(request.params['lon']),
                      float(request.params['lat']))
    elif 'geometry' in request.params:
        shape = loads(request.params['geometry'],
                      object_hook=GeoJSON.to_instance)
        shape = asShape(shape)
    if shape is None:
        return None
    column_epsg = _get_col_epsg(mapped_class, geom_attr)
    geom_attr = getattr(mapped_class, geom_attr)
    epsg = column_epsg if epsg is None else epsg
    if epsg != column_epsg:
        geom_attr = func.ST_Transform(geom_attr, epsg)
    geometry = from_shape(shape, srid=epsg)
    return func.ST_DWITHIN(geom_attr, geometry, tolerance)