def create_filter(request, mapped_class, geom_attr, **kwargs):
    """ Create MapFish default filter based on the request params.

    Arguments:

    request
        the request.

    mapped_class
        the SQLAlchemy mapped class.

    geom_attr
        the key of the geometry property as defined in the SQLAlchemy
        mapper. If you use ``declarative_base`` this is the name of
        the geometry attribute as defined in the mapped class.

    \\**kwargs
        additional arguments passed to ``create_geom_filter()``.
    """
    attr_filter = create_attr_filter(request, mapped_class)
    geom_filter = create_geom_filter(request, mapped_class, geom_attr,
                                     **kwargs)
    if geom_filter is None and attr_filter is None:
        return None
    if geom_filter is None:
        return attr_filter
    if attr_filter is None:
        return geom_filter
    return and_(geom_filter, attr_filter)