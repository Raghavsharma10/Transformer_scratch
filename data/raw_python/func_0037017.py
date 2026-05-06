def _scalar_pattern_uniform_op_left(func):
        """Decorator for operator overloading when ScalarPatternUniform is on 
        the left."""
        @wraps(func)
        def verif(self, patt):
            if isinstance(patt, ScalarPatternUniform):
                if self._dsphere.shape == patt._dsphere.shape:
                    return ScalarPatternUniform(func(self, self._dsphere,
                                                     patt._dsphere),
                                                doublesphere=True)
                else:
                    raise ValueError(err_msg['SP_sz_msmtch'] % \
                                            (self.nrows, self.ncols,
                                            patt.nrows, patt.ncols))
        
            elif isinstance(patt, numbers.Number):
                return ScalarPatternUniform(func(self, self._dsphere, patt),
                                            doublesphere=True)
            else:
                raise TypeError(err_msg['no_combi_SP'])
        return verif