def thresholds(self):
        """Threshold values of the response functions."""
        return numpy.array(
            sorted(self._key2float(key) for key in self._coefs), dtype=float)