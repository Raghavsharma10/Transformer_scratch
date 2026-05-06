def check_lazy_load_wegobject(f):
    '''
    Decorator function to lazy load a :class:`Wegobject`.
    '''
    def wrapper(*args):
        wegobject = args[0]
        if (
            wegobject._centroid is None or
            wegobject._bounding_box is None or
            wegobject._metadata is None
        ):
            log.debug('Lazy loading Wegobject %d', wegobject.id)
            wegobject.check_gateway()
            w = wegobject.gateway.get_wegobject_by_id(wegobject.id)
            wegobject._centroid = w._centroid
            wegobject._bounding_box = w._bounding_box
            wegobject._metadata = w._metadata
        return f(*args)
    return wrapper