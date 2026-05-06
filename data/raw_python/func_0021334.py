def predict(self, t):
        """Predict the smoothed function value at time t

        Parameters
        ----------
        t : array_like
            Times at which to predict the result

        Returns
        -------
        y : ndarray
            Smoothed values at time t
        """
        t = np.asarray(t)
        return self._predict(np.ravel(t)).reshape(t.shape)