def eigenvalues_samples(self):
        r""" Samples of the eigenvalues """
        res = np.empty((self.nsamples, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :] = self._sampled_hmms[i].eigenvalues
        return res