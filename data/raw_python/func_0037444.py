def get_negative(self, cls=None, **kwargs):
        """Returns a generator that generates negative cases by
        "each negative value in separate case" algorithm.
        """
        for attr, set_of_values in kwargs.iteritems():
            defaults = {key: kwargs[key][-1]["default"] for key in kwargs}
            defaults.pop(attr)
            for value in set_of_values[:-1]:
                case = cls() if cls else self._CasesClass()
                setattr(case, attr, value)
                for key in defaults:
                    setattr(case, key, defaults[key])
                yield case