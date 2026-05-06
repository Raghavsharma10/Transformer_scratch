def M(self, t, tips=None, gaps=None):
        """See docs for method in `Model` abstract base class."""
        assert isinstance(t, float) and t > 0, "Invalid t: {0}".format(t)
        with scipy.errstate(under='ignore'): # don't worry if some values 0
            if ('expD', t) not in self._cached:
                self._cached[('expD', t)] = scipy.exp(self.D * self.mu * t)
            expD = self._cached[('expD', t)]
            # swap axes to broadcast multiply D as diagonal matrix
            temp = scipy.ascontiguousarray((self.A.swapaxes(0, 1)
                    * expD).swapaxes(1, 0), dtype=float)
            M = broadcastMatrixMultiply(temp, self.Ainv)
            assert M.min() > -1e-3, "Overly negative M: {0}".format(M.min())
            M[M < 0] = 0.0
            if tips is None:
                return scipy.tile(M, (self.nsites, 1, 1))
            else:
                newM = scipy.zeros((len(tips), N_CODON))
                for i in range(len(tips)):
                    newM[i] =(M[0][:,tips[i]])
                if gaps is not None:
                    newM[gaps] = scipy.ones(N_CODON, dtype='float')
                return newM