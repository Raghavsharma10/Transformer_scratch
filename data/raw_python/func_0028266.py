def rate_limits(self):
        """Returns a list of rate limit details."""
        if not self._rate_limits:
            self._rate_limits = utilities.get_rate_limits(self.response)

        return self._rate_limits