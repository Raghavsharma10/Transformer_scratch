def check_lazy_load_perceel(f):
    '''
    Decorator function to lazy load a :class:`Perceel`.
    '''
    def wrapper(*args):
        perceel = args[0]
        if perceel._centroid is None or perceel._metadata is None:
            log.debug('Lazy loading Perceel %s', perceel.id)
            perceel.check_gateway()
            p = perceel.gateway.get_perceel_by_id(perceel.id)
            perceel._centroid = p._centroid
            perceel._metadata = p._metadata
        return f(*args)
    return wrapper