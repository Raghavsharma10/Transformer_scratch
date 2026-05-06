def eigenvectors_left_samples(self):
        r""" Samples of the left eigenvectors of the hidden transition matrix """
        res = np.empty((self.nsamples, self.nstates, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :, :] = self._sampled_hmms[i].eigenvectors_left
        return res