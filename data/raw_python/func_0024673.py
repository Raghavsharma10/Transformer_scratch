def _norm(self, value):
        """Normalize the input value into the fundamental unit for this prefix
type.

   :param number value: The input value to be normalized
   :raises ValueError: if the input value is not a type of real number
"""
        if isinstance(value, self.valid_types):
            self._byte_value = value * self._unit_value
            self._bit_value = self._byte_value * 8.0
        else:
            raise ValueError("Initialization value '%s' is of an invalid type: %s. "
                             "Must be one of %s" % (
                                 value,
                                 type(value),
                                 ", ".join(str(x) for x in self.valid_types)))