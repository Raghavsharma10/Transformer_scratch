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
                dM_param = broadcastMatrixMultiply(self.Prxy, Mt, alpha=alpha)
            else:
                dM_param = broadcastMatrixVectorMultiply(self.Prxy, Mt, alpha=alpha)
                if gaps is not None:
                    dM_param[gaps] = scipy.zeros(N_CODON, dtype='float')
            return dM_param

        paramval = getattr(self, param)
        if isinstance(paramval, float):
            paramisvec = False
        else:
            assert isinstance(paramval, numpy.ndarray) and paramval.ndim == 1
            paramisvec = True
            paramlength = paramval.shape[0]

        if ('expD', t) not in self._cached:
            self._cached[('expD', t)] = scipy.exp(self.D * self.mu * t)
        expD = self._cached[('expD', t)]

        if ('V', t) not in self._cached:
            if 'Dxx_Dyy' not in self._cached:
                Dyy = scipy.tile(self.D, (1, N_CODON)).reshape(
                        self.nsites, N_CODON, N_CODON)
                Dxx = scipy.array([Dyy[r].transpose() for r in
                        range(self.nsites)])
                self._cached['Dxx_Dyy'] = Dxx - Dyy
            Dxx_Dyy = self._cached['Dxx_Dyy']
            if 'Dxx_Dyy_lt_ALMOST_ZERO' not in self._cached:
                self._cached['Dxx_Dyy_lt_ALMOST_ZERO'] = scipy.fabs(
                        Dxx_Dyy) < ALMOST_ZERO
            Dxx_Dyy_lt_ALMOST_ZERO = self._cached['Dxx_Dyy_lt_ALMOST_ZERO']
            with scipy.errstate(divide='raise', under='ignore',
                    over='raise', invalid='ignore'):
                expDyy = scipy.tile(expD,(1, N_CODON)).reshape(
                        self.nsites, N_CODON, N_CODON)
                expDxx = scipy.array([expDyy[r].transpose() for r in
                        range(self.nsites)])
                V = (expDxx - expDyy) / Dxx_Dyy
            with scipy.errstate(under='ignore'): # OK if some values 0
                scipy.copyto(V, self.mu * t * expDxx, where=
                        Dxx_Dyy_lt_ALMOST_ZERO)
            self._cached[('V', t)] = V
        V = self._cached[('V', t)]

        with scipy.errstate(under='ignore'): # don't worry if some values 0
            if tips is None:
                if not paramisvec:
                    dM_param = broadcastMatrixMultiply(self.A,
                            broadcastMatrixMultiply(self.B[param]
                            * V, self.Ainv))
                else:
                    dM_param = scipy.ndarray((paramlength, self.nsites,
                            N_CODON, N_CODON), dtype='float')
                    for j in range(paramlength):
                        dM_param[j] = broadcastMatrixMultiply(self.A,
                                broadcastMatrixMultiply(self.B[param][j]
                                * V, self.Ainv))
            else:
                if not paramisvec:
                    dM_param = broadcastMatrixVectorMultiply(self.A,
                            broadcastGetCols(broadcastMatrixMultiply(
                            self.B[param] * V, self.Ainv), tips))
                else:
                    dM_param = scipy.ndarray((paramlength, self.nsites,
                            N_CODON), dtype='float')
                    for j in range(paramlength):
                        dM_param[j] = broadcastMatrixVectorMultiply(self.A,
                            broadcastGetCols(broadcastMatrixMultiply(
                            self.B[param][j] * V, self.Ainv), tips))
                if gaps is not None:
                    if not paramisvec:
                        dM_param[gaps] = scipy.zeros(N_CODON, dtype='float')
                    else:
                        dM_param[:, gaps] = scipy.zeros(N_CODON, dtype='float')
        return dM_param