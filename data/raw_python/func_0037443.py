def get_pairwise(self, cls=None, **kwargs):
        """Returns a generator that generates positive cases by
        "pairwise" algorithm.
        """
        for set_of_values in allpairs(kwargs.values()):
            case = cls() if cls else self._CasesClass()
            for attr, value in izip(kwargs.keys(), set_of_values):
                setattr(case, attr, value)
            yield case