def M(self, t, tips=None, gaps=None):
        """See docs for method in `Model` abstract base class."""
        assert isinstance(t, float) and t > 0, "Invalid t: {0}".format(t)
        with scipy.errstate(under='ignore'): # don't worry if some values 0
            if ('expD', t) not in self._cached:
                self._cached[('expD', t)] = scipy.exp(self.D * self.mu * t)
            expD = self._cached[('expD', t)]
            if tips is None:
                # swap axes to broadcast multiply D as diagonal matrix
                M = broadcastMatrixMultiply((self.A.swapaxes(0, 1) *
                        expD).swapaxes(1, 0), self.Ainv)
            else:
                M = broadcastMatrixVectorMultiply((self.A.swapaxes(0, 1)
                        * expD).swapaxes(1, 0), broadcastGetCols(
                        self.Ainv, tips))
                if gaps is not None:
                    M[gaps] = scipy.ones(N_CODON, dtype='float')
        #if M.min() < -0.01:
        #    warnings.warn("Large negative value in M(t) being set to 0. "
        #            "Value is {0}, t is {1}".format(M.min(), t))
        M[M < 0] = 0.0
        return M