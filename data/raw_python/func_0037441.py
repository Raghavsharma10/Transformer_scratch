def get_one(self, cls=None, **kwargs):
        """Returns a one case."""
        case = cls() if cls else self._CasesClass()
        for attr, value in kwargs.iteritems():
            setattr(case, attr, value)
        return case