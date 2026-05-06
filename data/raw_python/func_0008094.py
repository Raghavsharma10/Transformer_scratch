def stationary_distribution(self):
        r""" Compute stationary distribution of hidden states if possible.

        Raises
        ------
        ValueError if the HMM is not stationary

        """
        assert _tmatrix_disconnected.is_connected(self._Tij, strong=False), \
            'No unique stationary distribution because transition matrix is not connected'
        import msmtools.analysis as msmana
        return msmana.stationary_distribution(self._Tij)