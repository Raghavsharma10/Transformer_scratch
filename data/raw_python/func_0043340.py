def check_lazy_load_subadres(f):
    '''
    Decorator function to lazy load a :class:`Subadres`.
    '''
    def wrapper(*args):
        subadres = args[0]
        if (
            subadres._metadata is None or
            subadres.aard_id is None or
            subadres.huisnummer_id is None
        ):
            log.debug('Lazy loading Subadres %d', subadres.id)
            subadres.check_gateway()
            s = subadres.gateway.get_subadres_by_id(subadres.id)
            subadres._metadata = s._metadata
            subadres.aard_id = s.aard_id
            subadres.huisnummer_id = s.huisnummer_id
        return f(*args)
    return wrapper