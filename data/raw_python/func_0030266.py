def hours(start, end=None):
    """Iterate over the hours between the given datetime_tzs.

    Args:
      start: datetime_tz to start from.
      end: (Optional) Date to end at, if not given the iterator will never
           terminate.

    Returns:
      An iterator which generates datetime_tz objects a hour apart.
    """
    return iterate.between(start, datetime.timedelta(hours=1), end)