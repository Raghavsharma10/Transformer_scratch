def _set_shape(self, shape):
        """Private on purpose."""
        try:
            shape = (int(shape),)
        except TypeError:
            pass
        shp = list(shape)
        shp[0] = timetools.Period('366d')/self.simulationstep
        shp[0] = int(numpy.ceil(round(shp[0], 10)))
        getattr(self.fastaccess, self.name).ratios = numpy.zeros(
            shp, dtype=float)