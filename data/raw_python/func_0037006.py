def copy(self):
        """Make a deep copy of this object.
        
        Example::

            >>> c2 = c.copy()
        
        """
        vec = np.copy(self._vec)
        return ScalarCoefs(vec, self.nmax, self.mmax)