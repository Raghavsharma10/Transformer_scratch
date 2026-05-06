def check_lazy_load_gebouw(f):
    '''
    Decorator function to lazy load a :class:`Gebouw`.
    '''
    def wrapper(*args):
        gebouw = args[0]
        if (
            gebouw._methode_id is None or gebouw._geometrie is None or
            gebouw._metadata is None
        ):
            log.debug('Lazy loading Gebouw %d', gebouw.id)
            gebouw.check_gateway()
            g = gebouw.gateway.get_gebouw_by_id(gebouw.id)
            gebouw._methode_id = g._methode_id
            gebouw._geometrie = g._geometrie
            gebouw._metadata = g._metadata
        return f(*args)
    return wrapper