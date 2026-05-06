def retrieve_interval(self, start_time, end_time, compute_missing=False):
    """
    Return the results for `query_function` on every `bucket_width`
    time period between `start_time` and `end_time`.  Look for
    previously cached results to avoid recomputation.

    :param start_time: A datetime for the beginning of the range,
    aligned with `bucket_width`.
    :param end_time: A datetime for the end of the range, aligned with
    `bucket_width`.
    :param compute_missing: A boolean that, if True, will compute any
    non-cached results.
    """
    events = self._compute_buckets(start_time, end_time,
                                   compute_missing=compute_missing)

    for event in events:
      yield event