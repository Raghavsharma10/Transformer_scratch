def scalar_term(self, st):
        """Return a _ScalarTermS or _ScalarTermU from a string, to perform text and HTML substitutions"""
        if isinstance(st, binary_type):
            return _ScalarTermS(st, self._jinja_sub)
        elif isinstance(st, text_type):
            return _ScalarTermU(st, self._jinja_sub)
        elif st is None:
            return _ScalarTermU(u(''), self._jinja_sub)
        else:
            return st