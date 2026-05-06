def get_each_choice(self, cls=None, **kwargs):
        """Returns a generator that generates positive cases by
        "each choice" algorithm.
        """
        defaults = {attr: kwargs[attr][0] for attr in kwargs}
        for set_of_values in izip_longest(*kwargs.values()):
            case = cls() if cls else self._CasesClass()
            for attr, value in izip(kwargs.keys(), set_of_values):
                if value is None:
                    value = defaults[attr]
                setattr(case, attr, value)
            yield case