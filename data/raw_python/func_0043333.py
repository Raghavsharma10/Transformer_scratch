def check_lazy_load_straat(f):
    '''
    Decorator function to lazy load a :class:`Straat`.
    '''
    def wrapper(*args):
        straat = args[0]
        if (
            straat._namen is None or straat._metadata is None
        ):
            log.debug('Lazy loading Straat %d', straat.id)
            straat.check_gateway()
            s = straat.gateway.get_straat_by_id(straat.id)
            straat._namen = s._namen
            straat._metadata = s._metadata
        return f(*args)
    return wrapper