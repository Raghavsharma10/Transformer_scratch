def rate_limits(self):
        """Returns list of rate limit information from the response"""
        if not self._rate_limits:
            self._rate_limits = utilities.get_rate_limits(self._response)
        return self._rate_limits