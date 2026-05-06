def is_stationary(self):
        r""" Whether the MSM is stationary, i.e. whether the initial distribution is the stationary distribution
         of the hidden transition matrix. """
        # for disconnected matrices, the stationary distribution depends on the estimator, so we can't compute
        # it directly. Therefore we test whether the initial distribution is stationary.
        return np.allclose(np.dot(self._Pi, self._Tij), self._Pi)