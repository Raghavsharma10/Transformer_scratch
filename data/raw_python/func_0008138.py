def means_samples(self):
        r""" Samples of the Gaussian distribution means """
        res = np.empty((self.nsamples, self.nstates, self.dimension), dtype=config.dtype)
        for i in range(self.nsamples):
            for j in range(self.nstates):
                res[i, j, :] = self._sampled_hmms[i].means[j]
        return res