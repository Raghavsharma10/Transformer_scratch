def scan(self, t, dt=None, aggfunc=None):
        """
        Returns the spectrum from a specific time.

        Parameters
        ----------
        t : float
        dt : float
        """
        idx = (np.abs(self.index - t)).argmin()

        if dt is None:
            # only take the spectra at the nearest time
            mz_abn = self.values[idx, :].copy()
        else:
            # sum up all the spectra over a range
            en_idx = (np.abs(self.index - t - dt)).argmin()
            idx, en_idx = min(idx, en_idx), max(idx, en_idx)
            if aggfunc is None:
                mz_abn = self.values[idx:en_idx + 1, :].copy().sum(axis=0)
            else:
                mz_abn = aggfunc(self.values[idx:en_idx + 1, :].copy())
        if isinstance(mz_abn, scipy.sparse.spmatrix):
            mz_abn = mz_abn.toarray()[0]
        return Scan(self.columns, mz_abn)