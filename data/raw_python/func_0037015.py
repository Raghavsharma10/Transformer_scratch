def _vector_coef_op_left(func):
        """decorator for operator overloading when VectorCoef is on the
        left"""
        @wraps(func)
        def verif(self, vcoef):
            if isinstance(vcoef, VectorCoefs):
                if len(vcoef.scoef1._vec) == len(vcoef.scoef1._vec):
                    return VectorCoefs(func(self, self.scoef1._vec,
                                                  vcoef.scoef1._vec),
                                       func(self, self.scoef2._vec,
                                                  vcoef.scoef2._vec),
                                        self.nmax,
                                        self.mmax)
                else:
                    raise ValueError(err_msg['VC_sz_msmtch'] % \
                                    (self.nmax, self.mmax,
                                     vcoef.nmax, vcoef.mmax))
        
            elif isinstance(vcoef, numbers.Number):
                return VectorCoefs(func(self, self.scoef1._vec, vcoef),
                                   func(self, self.scoef2._vec, vcoef),
                                   self.nmax,
                                   self.mmax)
            else:
                raise TypeError(err_msg['no_combi_VC'])
        return verif