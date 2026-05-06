def initial_distribution_samples(self):
        r""" Samples of the initial distribution """
        res = np.empty((self.nsamples, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :] = self._sampled_hmms[i].stationary_distribution
        return res