def cache(self):
    """Call a user defined query and cache the results"""
    if not self._bucket_width or self._untrusted_time is None:
      raise ValueError('QueryCompute must be initialized with a bucket_width '
                       'and an untrusted_time in order to write to the cache.')

    now = datetime.datetime.now()
    untrusted_time = now - datetime.timedelta(seconds=self._untrusted_time)
    list(self._query_cache.compute_and_cache_missing_buckets(
        self._start_time,
        self._end_time,
        untrusted_time))