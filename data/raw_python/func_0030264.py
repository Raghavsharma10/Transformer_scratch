def between(start, delta, end=None):
    """Return an iterator between this date till given end point.

    Example usage:
      >>> d = datetime_tz.smartparse("5 days ago")
      2008/05/12 11:45
      >>> for i in d.between(timedelta(days=1), datetime_tz.now()):
      >>>    print i
      2008/05/12 11:45
      2008/05/13 11:45
      2008/05/14 11:45
      2008/05/15 11:45
      2008/05/16 11:45

    Args:
      start: The date to start at.
      delta: The interval to iterate with.
      end: (Optional) Date to end at. If not given the iterator will never
           terminate.

    Yields:
      datetime_tz objects.
    """
    toyield = start
    while end is None or toyield < end:
      yield toyield
      toyield += delta