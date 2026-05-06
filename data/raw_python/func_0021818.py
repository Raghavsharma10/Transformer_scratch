def cache_incr(self, key):
        """
        Non-atomic cache increment operation. Not optimal but
        consistent across different cache backends.
        """
        cache.set(key, cache.get(key, 0) + 1, self.expire_after())