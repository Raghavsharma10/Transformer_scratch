def timescales_samples(self):
        r""" Samples of the timescales """
        res = np.empty((self.nsamples, self.nstates-1), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :] = self._sampled_hmms[i].timescales
        return res