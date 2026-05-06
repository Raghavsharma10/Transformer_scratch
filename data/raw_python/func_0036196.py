def _bucket_events(self, event_iterable):
    """
    Convert an iterable of events into an iterable of lists of events
    per bucket.
    """

    current_bucket_time = None
    current_bucket_events = None
    for event in event_iterable:
      event_bucket_time = self._bucket_time(event[TIMESTAMP_FIELD])
      if current_bucket_time is None or current_bucket_time < event_bucket_time:
        if current_bucket_events is not None:
          yield current_bucket_events
        current_bucket_time = event_bucket_time
        current_bucket_events = []
      current_bucket_events.append(event)
    if current_bucket_events is not None and current_bucket_events != []:
      yield current_bucket_events