def sigmas_samples(self):
        r""" Samples of the Gaussian distribution standard deviations """
        res = np.empty((self.nsamples, self.nstates, self.dimension), dtype=config.dtype)
        for i in range(self.nsamples):
            for j in range(self.nstates):
                res[i, j, :] = self._sampled_hmms[i].sigmas[j]
        return res