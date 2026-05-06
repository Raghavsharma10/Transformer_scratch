def _scalar_pattern_uniform_op_right(func):
        """Decorator for operator overloading when ScalarPatternUniform is on
        the right."""
        @wraps(func)
        def verif(self, patt):
            if isinstance(patt, numbers.Number):
                return ScalarPatternUniform(func(self, self._dsphere, patt),
                                   doublesphere=True)
            else:
                raise TypeError(err_msg['no_combi_SP'])
        return verif