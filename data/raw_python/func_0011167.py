def get_backoff_time(self):
        """ Formula for computing the current backoff

        :rtype: float
        """
        if self._observed_errors <= 1:
            return 0

        backoff_value = self.backoff_factor * (2 ** (self._observed_errors - 1))
        return min(self.BACKOFF_MAX, backoff_value)