def check_lazy_load_sectie(f):
    '''
    Decorator function to lazy load a :class:`Sectie`.
    '''

    def wrapper(self):
        sectie = self
        if (getattr(sectie, '_%s' % f.__name__, None) is None):
            log.debug('Lazy loading Sectie %s in Afdeling %d', sectie.id, sectie.afdeling.id)
            sectie.check_gateway()
            s = sectie.gateway.get_sectie_by_id_and_afdeling(
                sectie.id, sectie.afdeling.id
            )
            sectie._centroid = s._centroid
            sectie._bounding_box = s._bounding_box
        return f(self)

    return wrapper