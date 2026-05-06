def check_lazy_load_adrespositie(f):
    '''
    Decorator function to lazy load a :class:`Adrespositie`.
    '''
    def wrapper(*args):
        adrespositie = args[0]
        if (
            adrespositie._geometrie is None or
            adrespositie._aard is None or
            adrespositie._metadata is None
        ):
            log.debug('Lazy loading Adrespositie %d', adrespositie.id)
            adrespositie.check_gateway()
            a = adrespositie.gateway.get_adrespositie_by_id(adrespositie.id)
            adrespositie._geometrie = a._geometrie
            adrespositie.aard_id = a.aard_id
            adrespositie._metadata = a._metadata
        return f(*args)
    return wrapper