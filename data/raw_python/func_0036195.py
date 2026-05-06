def _bucket_time(self, event_time):
    """
    The seconds since epoch that represent a computed bucket.

    An event bucket is the time of the earliest possible event for
    that `bucket_width`.  Example: if `bucket_width =
    timedelta(minutes=10)`, bucket times will be the number of seconds
    since epoch at 12:00, 12:10, ...  on each day.
    """
    event_time = kronos_time_to_epoch_time(event_time)
    return event_time - (event_time % self._bucket_width)