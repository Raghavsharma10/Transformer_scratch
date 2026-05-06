def check_lazy_load_terreinobject(f):
    '''
    Decorator function to lazy load a :class:`Terreinobject`.
    '''
    def wrapper(*args):
        terreinobject = args[0]
        if (
            terreinobject._centroid is None or
            terreinobject._bounding_box is None or
            terreinobject._metadata is None
        ):
            log.debug('Lazy loading Terreinobject %s', terreinobject.id)
            terreinobject.check_gateway()
            t = terreinobject.gateway.get_terreinobject_by_id(terreinobject.id)
            terreinobject._centroid = t._centroid
            terreinobject._bounding_box = t._bounding_box
            terreinobject._metadata = t._metadata
        return f(*args)
    return wrapper