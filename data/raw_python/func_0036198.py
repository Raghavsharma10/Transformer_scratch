def compute_and_cache_missing_buckets(self, start_time, end_time,
                                        untrusted_time, force_recompute=False):
    """
    Return the results for `query_function` on every `bucket_width`
    time period between `start_time` and `end_time`.  Look for
    previously cached results to avoid recomputation.  For any buckets
    where all events would have occurred before `untrusted_time`,
    cache the results.

    :param start_time: A datetime for the beginning of the range,
    aligned with `bucket_width`.
    :param end_time: A datetime for the end of the range, aligned with
    `bucket_width`.
    :param untrusted_time: A datetime after which to not trust that
    computed data is stable.  Any buckets that overlap with or follow
    this untrusted_time will not be cached.
    :param force_recompute: A boolean that, if True, will force
    recompute and recaching of even previously cached data.
    """
    if untrusted_time and not untrusted_time.tzinfo:
      untrusted_time = untrusted_time.replace(tzinfo=tzutc())

    events = self._compute_buckets(start_time, end_time, compute_missing=True,
                                   cache=True, untrusted_time=untrusted_time,
                                   force_recompute=force_recompute)

    for event in events:
      yield event