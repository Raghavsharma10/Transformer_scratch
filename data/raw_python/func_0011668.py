def angle(self, deg=False):
        """Return the angle of the complex argument.

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
        return Timeseries(np.angle(self, deg=deg), self.tspan, self.labels)