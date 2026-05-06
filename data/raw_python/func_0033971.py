def fit(self, P):
        """Fit the diagonal matrices in Sinkhorn Knopp's algorithm

        Parameters
        ----------
        P : 2d array-like
        Must be a square non-negative 2d array-like object, that
        is convertible to a numpy array. The matrix must not be
        equal to 0 and it must have total support for the algorithm
        to converge.

        Returns
        -------
        A double stochastic matrix.

        """
        P = np.asarray(P)
        assert np.all(P >= 0)
        assert P.ndim == 2
        assert P.shape[0] == P.shape[1]

        N = P.shape[0]
        max_thresh = 1 + self._epsilon
        min_thresh = 1 - self._epsilon

        # Initialize r and c, the diagonals of D1 and D2
        # and warn if the matrix does not have support.
        r = np.ones((N, 1))
        pdotr = P.T.dot(r)
        total_support_warning_str = (
            "Matrix P must have total support. "
            "See documentation"
        )
        if not np.all(pdotr != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        c = 1 / pdotr
        pdotc = P.dot(c)
        if not np.all(pdotc != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        r = 1 / pdotc
        del pdotr, pdotc

        P_eps = np.copy(P)
        while np.any(np.sum(P_eps, axis=1) < min_thresh) \
                or np.any(np.sum(P_eps, axis=1) > max_thresh) \
                or np.any(np.sum(P_eps, axis=0) < min_thresh) \
                or np.any(np.sum(P_eps, axis=0) > max_thresh):

            c = 1 / P.T.dot(r)
            r = 1 / P.dot(c)

            self._D1 = np.diag(np.squeeze(r))
            self._D2 = np.diag(np.squeeze(c))
            P_eps = self._D1.dot(P).dot(self._D2)

            self._iterations += 1

            if self._iterations >= self._max_iter:
                self._stopping_condition = "max_iter"
                break

        if not self._stopping_condition:
            self._stopping_condition = "epsilon"

        self._D1 = np.diag(np.squeeze(r))
        self._D2 = np.diag(np.squeeze(c))
        P_eps = self._D1.dot(P).dot(self._D2)

        return P_eps