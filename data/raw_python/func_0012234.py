def rate_limit_wait(self):
        """
        Sleep if rate limiting is required based on current time and last
        query.

        """
        if self._rate_limit_dt and self._last_query is not None:
            dt = time.time() - self._last_query
            wait = self._rate_limit_dt - dt
            if wait > 0:
                time.sleep(wait)