def check_lazy_load_perceel(f):
    '''
    Decorator function to lazy load a :class:`Perceel`.
    '''

    def wrapper(self):
        perceel = self
        if (getattr(perceel, '_%s' % f.__name__, None) is None):
            log.debug(
                'Lazy loading Perceel %s in Sectie %s in Afdeling %d',
                perceel.id,
                perceel.sectie.id,
                perceel.sectie.afdeling.id
            )
            perceel.check_gateway()
            p = perceel.gateway.get_perceel_by_id_and_sectie(
                perceel.id,
                perceel.sectie
            )
            perceel._centroid = p._centroid
            perceel._bounding_box = p._bounding_box
            perceel._capatype = p._capatype
            perceel._cashkey = p._cashkey
        return f(self)

    return wrapper