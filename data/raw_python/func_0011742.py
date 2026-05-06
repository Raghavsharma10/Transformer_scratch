def angle(self, deg=False):
        """Return the angle of a complex Timeseries

        Args:
          deg (bool, optional):
            Return angle in degrees if True, radians if False (default).

        Returns:
          angle (Timeseries):
            The counterclockwise angle from the positive real axis on
            the complex plane, with dtype as numpy.float64.
        """
        if self.dtype.str[1] != 'c':
            warnings.warn('angle() is intended for complex-valued timeseries',
                          RuntimeWarning, 1)
        da = distob.vectorize(np.angle)(self, deg)
        return _dts_from_da(da, self.tspan, self.labels)