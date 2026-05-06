def _vector_pattern_uniform_op_left(func):
        """decorator for operator overloading when VectorPatternUniform is on 
        the left"""
        @wraps(func)
        def verif(self, patt):
            if isinstance(patt, TransversePatternUniform):
                if self._tdsphere.shape == patt._tdsphere.shape:
                    return TransversePatternUniform(func(self, self._tdsphere,
                                                     patt._tdsphere),
                                                func(self, self._pdsphere,
                                                     patt._pdsphere),
                                                doublesphere=True)
                else:
                    raise ValueError(err_msg['VP_sz_msmtch'] % \
                                            (self.nrows, self.ncols,
                                            patt.nrows, patt.ncols))
        
            elif isinstance(patt, numbers.Number):
                return TransversePatternUniform(func(self, self._tdsphere, patt),
                                            func(self, self._pdsphere, patt),
                                            doublesphere=True)
            else:
                raise TypeError(err_msg['no_combi_VP'])
        return verif