def window(self, vec):
        """Apply a window to the coefficients defined by *vec*. *vec* must
        have length *nmax* + 1.  This is good way to filter the pattern by
        windowing in the coefficient domain.

        Example::

            >>> vec = numpy.linspace(0, 1, c.nmax + 1)
            >>> c.window(vec)

        Args:
          vec (numpy.array): Vector of values to apply in the n direction of
          the data. Has length *nmax* + 1.

        Returns:
          Nothing, applies the window to the data in place.

        """

        slce = slice(None, None, None)
        
        self.__setitem__((slce, 0), self.__getitem__((slce, 0)) * vec)  
        for m in xrange(1, self.mmax + 1):
            self.__setitem__((slce, -m), self.__getitem__((slce, -m)) * vec[m:])
            self.__setitem__((slce, m), self.__getitem__((slce, m)) * vec[m:])