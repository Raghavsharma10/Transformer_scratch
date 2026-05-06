def get(self, *raw_args, **raw_kwargs):
        """
        Return the data for this function (using the cache if possible).

        This method is not intended to be overidden
        """
        # We pass args and kwargs through a filter to allow them to be
        # converted into values that can be pickled.
        args = self.prepare_args(*raw_args)
        kwargs = self.prepare_kwargs(**raw_kwargs)

        # Build the cache key and attempt to fetch the cached item
        key = self.key(*args, **kwargs)
        item = self.cache.get(key)

        call = Call(args=raw_args, kwargs=raw_kwargs)

        if item is None:
            # Cache MISS - we can either:
            # a) fetch the data immediately, blocking execution until
            #    the fetch has finished, or
            # b) trigger an async refresh and return an empty result
            if self.should_missing_item_be_fetched_synchronously(*args, **kwargs):
                logger.debug(("Job %s with key '%s' - cache MISS - running "
                              "synchronous refresh"),
                             self.class_path, key)
                result = self.refresh(*args, **kwargs)
                return self.process_result(
                    result, call=call, cache_status=self.MISS, sync_fetch=True)

            else:
                logger.debug(("Job %s with key '%s' - cache MISS - triggering "
                              "async refresh and returning empty result"),
                             self.class_path, key)
                # To avoid cache hammering (ie lots of identical tasks
                # to refresh the same cache item), we reset the cache with an
                # empty result which will be returned until the cache is
                # refreshed.
                result = self.empty()
                self.store(key, self.timeout(*args, **kwargs), result)
                self.async_refresh(*args, **kwargs)
                return self.process_result(
                    result, call=call, cache_status=self.MISS,
                    sync_fetch=False)

        expiry, data = item
        delta = time.time() - expiry
        if delta > 0:
            # Cache HIT but STALE expiry - we can either:
            # a) fetch the data immediately, blocking execution until
            #    the fetch has finished, or
            # b) trigger a refresh but allow the stale result to be
            #    returned this time.  This is normally acceptable.
            if self.should_stale_item_be_fetched_synchronously(
                    delta, *args, **kwargs):
                logger.debug(
                    ("Job %s with key '%s' - STALE cache hit - running "
                     "synchronous refresh"),
                    self.class_path, key)
                result = self.refresh(*args, **kwargs)
                return self.process_result(
                    result, call=call, cache_status=self.STALE,
                    sync_fetch=True)

            else:
                logger.debug(
                    ("Job %s with key '%s' - STALE cache hit - triggering "
                     "async refresh and returning stale result"),
                    self.class_path, key)
                # We replace the item in the cache with a 'timeout' expiry - this
                # prevents cache hammering but guards against a 'limbo' situation
                # where the refresh task fails for some reason.
                timeout = self.timeout(*args, **kwargs)
                self.store(key, timeout, data)
                self.async_refresh(*args, **kwargs)
                return self.process_result(
                    data, call=call, cache_status=self.STALE, sync_fetch=False)
        else:
            logger.debug("Job %s with key '%s' - cache HIT", self.class_path, key)
            return self.process_result(data, call=call, cache_status=self.HIT)