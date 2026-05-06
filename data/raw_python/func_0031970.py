def _contains_value(self, value):
        """Helper function for __contains__ to check a single value is contained within the interval"""
        g = operator.gt if self._lower is self.OPEN else operator.ge
        l = operator.lt if self._upper is self.OPEN else operator.le
        return g(value, self.lower_value) and l(value, self._upper_value)