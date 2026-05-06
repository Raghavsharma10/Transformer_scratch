def asdatetime(self, naive=True):
    """Return this datetime_tz as a datetime object.

    Args:
      naive: Return *without* any tz info.

    Returns:
      This datetime_tz as a datetime object.
    """
    args = list(self.timetuple()[0:6])+[self.microsecond]
    if not naive:
      args.append(self.tzinfo)
    return datetime.datetime(*args)