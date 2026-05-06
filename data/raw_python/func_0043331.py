def check_lazy_load_gewest(f):
    '''
    Decorator function to lazy load a :class:`Gewest`.
    '''
    def wrapper(self):
        gewest = self
        attribute = 'namen' if f.__name__ == 'naam' else f.__name__
        if (getattr(gewest, '_%s' % attribute, None) is None):
            log.debug('Lazy loading Gewest %d', gewest.id)
            gewest.check_gateway()
            g = gewest.gateway.get_gewest_by_id(gewest.id)
            gewest._namen = g._namen
            gewest._centroid = g._centroid
            gewest._bounding_box = g._bounding_box
        return f(self)
    return wrapper