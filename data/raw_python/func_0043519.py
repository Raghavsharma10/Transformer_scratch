def check_lazy_load_afdeling(f):
    '''
    Decorator function to lazy load a :class:`Afdeling`.
    '''

    def wrapper(self):
        afdeling = self
        if (getattr(afdeling, '_%s' % f.__name__, None) is None):
            log.debug('Lazy loading Afdeling %d', afdeling.id)
            afdeling.check_gateway()
            a = afdeling.gateway.get_kadastrale_afdeling_by_id(afdeling.id)
            afdeling._naam = a._naam
            afdeling._gemeente = a._gemeente
            afdeling._centroid = a._centroid
            afdeling._bounding_box = a._bounding_box
        return f(self)

    return wrapper