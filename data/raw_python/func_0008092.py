def update(self, Pi, Tij):
        r""" Updates the transition matrix and recomputes all derived quantities """
        from msmtools import analysis as msmana

        # update transition matrix by copy
        self._Tij = np.array(Tij)
        assert msmana.is_transition_matrix(self._Tij), 'Given transition matrix is not a stochastic matrix'
        assert self._Tij.shape[0] == self._nstates, 'Given transition matrix has unexpected number of states '
        # reset spectral decomposition
        self._spectral_decomp_available = False

        # check initial distribution
        assert np.all(Pi >= 0), 'Given initial distribution contains negative elements.'
        assert np.any(Pi > 0), 'Given initial distribution is zero'
        self._Pi = np.array(Pi) / np.sum(Pi)