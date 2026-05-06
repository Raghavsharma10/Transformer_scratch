def size(self):
        """Total number of coefficients in the ScalarCoefs structure.

        Example::

            >>> sz  = c.size
            >>> N = c.nmax + 1
            >>> L = N+ c.mmax * (2 * N - c.mmax - 1);
            >>> assert sz == L
        """
        N = self.nmax + 1;
        NC = N + self.mmax * (2 * N - self.mmax - 1);
        assert NC == len(self._vec)
        return NC