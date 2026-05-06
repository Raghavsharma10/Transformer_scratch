def copy(self):
        """Make a deep copy of this object.
        
        Example::

            >>> c2 = c.copy()
        
        """
        vec1 = np.copy(self.scoef1._vec)
        vec2 = np.copy(self.scoef2._vec)
        return VectorCoefs(vec1, vec2, self.nmax, self.mmax)