def now(cls, tzinfo=None):
    """[tz] -> new datetime with tz's local day and time."""
    obj = cls.utcnow()
    if tzinfo is None:
      tzinfo = localtz()
    return obj.astimezone(tzinfo)