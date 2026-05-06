def combine(cls, date, time, tzinfo=None):
    """date, time, [tz] -> datetime with same date and time fields."""
    if tzinfo is None:
      tzinfo = localtz()
    return cls(datetime.datetime.combine(date, time), tzinfo)