def check_lazy_load_huisnummer(f):
    '''
    Decorator function to lazy load a :class:`Huisnummer`.
    '''
    def wrapper(*args):
        huisnummer = args[0]
        if (
            huisnummer._metadata is None
        ):
            log.debug('Lazy loading Huisnummer %d', huisnummer.id)
            huisnummer.check_gateway()
            h = huisnummer.gateway.get_huisnummer_by_id(huisnummer.id)
            huisnummer._metadata = h._metadata
        return f(*args)
    return wrapper