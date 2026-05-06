def _scalar_coef_op_right(func):
        """decorator for operator overloading when ScalarCoef is on the
        right"""
        @wraps(func)
        def verif(self, scoef):
            if isinstance(scoef, numbers.Number):
                return ScalarCoefs(func(self, self._vec, scoef),
                                   self.nmax, self.mmax)
            else:
                raise TypeError(err_msg['no_combi_SC'])
        return verif