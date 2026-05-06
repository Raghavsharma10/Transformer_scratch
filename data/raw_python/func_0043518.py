def check_lazy_load_gemeente(f):
    '''
    Decorator function to lazy load a :class:`Gemeente`.
    '''

    def wrapper(self):
        gemeente = self
        if (getattr(gemeente, '_%s' % f.__name__, None) is None):
            log.debug('Lazy loading Gemeente %d', gemeente.id)
            gemeente.check_gateway()
            g = gemeente.gateway.get_gemeente_by_id(gemeente.id)
            gemeente._naam = g._naam
            gemeente._centroid = g._centroid
            gemeente._bounding_box = g._bounding_box
        return f(self)

    return wrapper