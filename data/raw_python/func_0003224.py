async def limit(self, use = 1):
        """
        Acquire "resources", wait until enough "resources" are acquired. For each loop,
        `limit` number of "resources" are permitted.
        
        :param use: number of "resouces" to be used.
        
        :return: True if is limited
        """
        c = self._counter
        self._counter = c + use
        if self._task is None:
            self._task = self._container.subroutine(self._limiter_task(), False)
        if c >= self._bottom_line:
            # Limited
            await RateLimitingEvent.createMatcher(self, c // self._limit)
            return True
        else:
            return False