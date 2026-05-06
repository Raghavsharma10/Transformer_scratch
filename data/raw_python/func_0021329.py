def get_value(self, merge=True, createfunc=None,
                  expiration_time=None, ignore_expiration=False):
        """
        Return the value from the cache for this query.
        """
        cache, cache_key = self._get_cache_plus_key()

        # ignore_expiration means, if the value is in the cache
        # but is expired, return it anyway.   This doesn't make sense
        # with createfunc, which says, if the value is expired, generate
        # a new value.
        assert not ignore_expiration or not createfunc, \
            "Can't ignore expiration and also provide createfunc"

        if ignore_expiration or not createfunc:
            cached_value = cache.get(cache_key,
                                     expiration_time=expiration_time,
                                     ignore_expiration=ignore_expiration)
        else:
            cached_value = cache.get(cache_key)
            if not cached_value:
                cached_value = createfunc()
                cache.set(cache_key, cached_value, timeout=expiration_time)

        if cached_value and merge:
            cached_value = self.merge_result(cached_value, load=False)

        return cached_value