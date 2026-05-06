def _error_and_gradient(self, x):
        """Compute the error and the gradient.

        This is the function optimized by :obj:`scipy.optimize.minimize`.

        Args:
            x (`array-like`): [`m` * `n`, ] matrix.

        Returns:
            `tuple`: containing:

                - Error (`float`)
                - Gradient (`np.array`) [`m`, `n`]
        """
        coords = x.reshape((self.m, self.n))
        d = squareform(pdist(coords))
        diff = self.D - d
        error = self._error(diff)
        gradient = self._gradient(diff, d, coords)
        return error, gradient.ravel()