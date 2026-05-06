def _vector_coef_op_right(func):
        """decorator for operator overloading when VectorCoefs is on the
        right"""
        @wraps(func)
        def verif(self, vcoef):
            if isinstance(vcoef, numbers.Number):
                return VectorCoefs(func(self, self.scoef1._vec, vcoef),
                                   func(self, self.scoef2._vec, vcoef),
                                   self.nmax, self.mmax)
            else:
                raise TypeError(err_msg['no_combi_VC'])
        return verif