def _load_int(self):
        """Load internal data from file and return it."""
        values = numpy.fromfile(self.filepath_int)
        if self.NDIM > 0:
            values = values.reshape(self.seriesshape)
        return values