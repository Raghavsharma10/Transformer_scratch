def _norm(self, value):
        """Normalize the input value into the fundamental unit for this prefix
type"""
        self._bit_value = value * self._unit_value
        self._byte_value = self._bit_value / 8.0