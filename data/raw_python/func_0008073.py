def transition_matrix_samples(self):
        r""" Samples of the transition matrix """
        res = np.empty((self.nsamples, self.nstates, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :, :] = self._sampled_hmms[i].transition_matrix
        return res