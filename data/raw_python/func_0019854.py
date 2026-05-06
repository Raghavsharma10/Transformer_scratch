def solve(self, b):
        """
        Solve linear equation A x = b for x

        Parameters
        ----------
        b : ndarray
            Right-hand side of the matrix equation. Can be vector or a matrix.

        Returns
        -------
        x : ndarray
            Solution to the matrix equation

        """
        if isspmatrix(b):
            b = b.toarray()

        if b.shape[0] != self._A.shape[1]:
            raise ValueError("Shape of b is not compatible with that of A")

        b_arr = asarray(b, dtype=self._A.dtype).reshape(b.shape[0], -1)
        x = np.zeros((self._A.shape[0], b_arr.shape[1]), dtype=self._A.dtype)
        for j in range(b_arr.shape[1]):
            x[:,j] = self.umf.solve(UMFPACK_A, self._A, b_arr[:,j], autoTranspose=True)
        return x.reshape((self._A.shape[0],) + b.shape[1:])