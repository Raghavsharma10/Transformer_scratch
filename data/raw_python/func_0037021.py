def _vector_pattern_uniform_op_right(func):
        """decorator for operator overloading when VectorPatternUniform is on
        the right"""
        @wraps(func)
        def verif(self, patt):
            if isinstance(patt, numbers.Number):
                return TransversePatternUniform(func(self, self._tdsphere, patt),
                                            func(self, self._pdsphere, patt),
                                            doublesphere=True)
            else:
                raise TypeError(err_msg['no_combi_VP'])
        return verif