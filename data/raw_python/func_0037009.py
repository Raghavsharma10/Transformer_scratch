def _array_2d_repr(self):
        """creates a 2D array that has nmax + 1 rows and 2*mmax + 1 columns
        and provides a representation for the coefficients that makes 
        plotting easier"""

        sc_array = np.zeros((self.nmax + 1, 2 * self.mmax + 1),
                            dtype=np.complex128)

        lst = self._reshape_n_vecs()
        sc_array[0:self.nmax + 1, self.mmax] = lst[0]
        for m in xrange(1, self.mmax + 1):
            sc_array[m:self.nmax + 1, self.mmax - m] = lst[2 * m - 1]
            sc_array[m:self.nmax + 1, self.mmax + m] = lst[2 * m]

        return sc_array