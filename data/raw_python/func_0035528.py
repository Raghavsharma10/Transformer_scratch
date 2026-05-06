def dM(self, t, param, Mt, tips=None, gaps=None):
        """See docs for method in `Model` abstract base class."""
        assert isinstance(t, float) and t > 0, "Invalid t: {0}".format(t)
        assert (param == 't') or (param in self.freeparams), (
                "Invalid param: {0}".format(param))

        if Mt is None:
            Mt = self.M(t, tips=tips, gaps=gaps)

        if (param == 'mu') or (param == 't'):
            if param == 'mu':
                alpha = t
            else:
                alpha = self.mu
            if tips is None:
                dM_param = scipy.tile(broadcastMatrixMultiply(self.Pxy,
                        scipy.tile(Mt[0], (1, 1, 1)), alpha=alpha),
                        (self.nsites, 1, 1))
            else:
                #Pxy is tiled over the number of sites
                dM_param = broadcastMatrixVectorMultiply(scipy.tile(self.Pxy[0],
                        (self.nsites, 1, 1)), Mt, alpha=alpha)
                if gaps is not None:
                    dM_param[gaps] = scipy.zeros(N_CODON, dtype='float')
            return dM_param

        paramval = getattr(self, param)
        assert isinstance(paramval, float), "All params should be floats"

        if ('expD', t) not in self._cached:
            self._cached[('expD', t)] = scipy.exp(self.D * self.mu * t)
        expD = self._cached[('expD', t)]

        if ('V', t) not in self._cached:
            if 'Dxx_Dyy' not in self._cached:
                Dyy = scipy.tile(self.D, (1, N_CODON)).reshape(
                        1, N_CODON, N_CODON)
                Dxx = scipy.array([Dyy[r].transpose() for r in
                        range(1)])
                self._cached['Dxx_Dyy'] = Dxx - Dyy
            Dxx_Dyy = self._cached['Dxx_Dyy']
            if 'Dxx_Dyy_lt_ALMOST_ZERO' not in self._cached:
                self._cached['Dxx_Dyy_lt_ALMOST_ZERO'] = scipy.fabs(
                        Dxx_Dyy) < ALMOST_ZERO
            Dxx_Dyy_lt_ALMOST_ZERO = self._cached['Dxx_Dyy_lt_ALMOST_ZERO']
            with scipy.errstate(divide='raise', under='ignore',
                    over='raise', invalid='ignore'):
                expDyy = scipy.tile(expD, (1, N_CODON)).reshape(
                        1, N_CODON, N_CODON)
                expDxx = scipy.array([expDyy[r].transpose() for r in
                        range(1)])
                V = (expDxx - expDyy) / Dxx_Dyy
            with scipy.errstate(under='ignore'): # OK if some values 0
                scipy.copyto(V, self.mu * t * expDxx, where=
                        Dxx_Dyy_lt_ALMOST_ZERO)
            self._cached[('V', t)] = V
        V = self._cached[('V', t)]

        with scipy.errstate(under='ignore'): # don't worry if some values 0
            dM_param = broadcastMatrixMultiply(self.A,
                        broadcastMatrixMultiply(self.B[param]
                        * V, self.Ainv))
            if tips is None:
                return scipy.tile(dM_param, (self.nsites, 1, 1))
            else:
                newdM_param = scipy.zeros((len(tips), N_CODON))
                for i in range(len(tips)):
                    newdM_param[i] =(dM_param[0][:,tips[i]])
                if gaps is not None:
                    newdM_param[gaps] = scipy.zeros(N_CODON, dtype='float')
                return newdM_param