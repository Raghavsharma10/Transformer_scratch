def compute(self, use_cache=True):
    """Call a user defined query and return events with optional help from
    the cache.

    :param use_cache: Specifies whether the cache should be used when possible
    """
    if use_cache:
      if not self._bucket_width:
        raise ValueError('QueryCompute must be initialized with a bucket_width'
                         ' to use caching features.')
      return list(self._query_cache.retrieve_interval(self._start_time,
                                                      self._end_time,
                                                      compute_missing=True))
    else:
      if self._metis:
        return self._run_metis(self._start_time, self._end_time)
      else:
        return self._run_query(self._start_time, self._end_time)