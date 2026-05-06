def _rate_limited():
        """A rate limit decorator for limiting the number of times the
        decorated :class:`Miner` method can be called. Limits are set in
        :attr:`Miner._max_per_second`.
        """
        def decorate(func):
            def rate_limited_func(self, *args, **kwargs):
                elapsed = time.monotonic() - self._last_time_called
                self.left_to_wait = self._min_interval - elapsed
                if self.left_to_wait > 0:
                    time.sleep(self.left_to_wait)
                func(self, *args, **kwargs)
                self._last_time_called = time.monotonic()
                yield from func(self, *args, **kwargs)
            return rate_limited_func
        return decorate