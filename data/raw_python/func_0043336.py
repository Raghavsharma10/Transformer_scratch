def check_lazy_load_wegsegment(f):
    '''
    Decorator function to lazy load a :class:`Wegsegment`.
    '''
    def wrapper(*args):
        wegsegment = args[0]
        if (
            wegsegment._methode_id is None or
            wegsegment._geometrie is None or
            wegsegment._metadata is None
        ):
            log.debug('Lazy loading Wegsegment %d', wegsegment.id)
            wegsegment.check_gateway()
            w = wegsegment.gateway.get_wegsegment_by_id(wegsegment.id)
            wegsegment._methode_id = w._methode_id
            wegsegment._geometrie = w._geometrie
            wegsegment._metadata = w._metadata
        return f(*args)
    return wrapper