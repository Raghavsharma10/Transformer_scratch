def _cached_results(self, start_time, end_time):
    """
    Retrieves cached results for any bucket that has a single cache entry.

    If a bucket has two cache entries, there is a chance that two
    different writers previously computed and cached a result since
    Kronos has no transaction semantics.  While it might be safe to
    return one of the cached results if there are multiple, we
    currently do the safe thing and pretend we have no previously
    computed data for this bucket.
    """
    cached_buckets = self._bucket_events(
      self._client.get(self._scratch_stream, start_time, end_time,
                       namespace=self._scratch_namespace))
    for bucket_events in cached_buckets:
      # If we have multiple cache entries for the same bucket, pretend
      # we have no results for that bucket.
      if len(bucket_events) == 1:
        first_result = bucket_events[0]
        yield (kronos_time_to_epoch_time(first_result[TIMESTAMP_FIELD]),
               first_result[QueryCache.CACHE_KEY])