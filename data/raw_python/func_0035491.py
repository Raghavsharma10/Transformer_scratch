def PARAMLIMITS(self, value):
        """Set new `PARAMLIMITS` dictionary."""
        assert set(value.keys()) == set(self.PARAMLIMITS.keys()), "The \
                new parameter limits are not defined for the same set \
                of parameters as before."
        for param in value.keys():
            assert value[param][0] < value[param][1], "The new \
                    minimum value for {0}, {1}, is equal to or \
                    larger than the new maximum value, {2}"\
                    .format(param, value[param][0], value[param][1])
        self._PARAMLIMITS = value.copy()