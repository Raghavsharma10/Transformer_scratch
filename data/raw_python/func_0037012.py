def _scalar_coef_op_left(func):
        """decorator for operator overloading when ScalarCoef is on the
        left"""
        @wraps(func)
        def verif(self, scoef):
            if isinstance(scoef, ScalarCoefs):
                if len(self._vec) == len(scoef._vec):
                    return ScalarCoefs(func(self, self._vec, scoef._vec),
                                        self.nmax,
                                        self.mmax)
                else:
                    raise ValueError(err_msg['SC_sz_msmtch'] % \
                                    (self.nmax, self.mmax,
                                     scoef.nmax, scoef.mmax))
        
            elif isinstance(scoef, numbers.Number):
                return ScalarCoefs(func(self, self._vec, scoef), self.nmax,
                                   self.mmax)
            else:
                raise TypeError(err_msg['no_combi_SC'])
        return verif