def localize(dt, force_to_local=True):
  """Localize a datetime to the local timezone.

  If dt is naive, returns the same datetime with the local timezone, otherwise
  uses astimezone to convert.

  Args:
    dt: datetime object.
    force_to_local: Force all results to be in local time.

  Returns:
    A datetime_tz object.
  """
  if not isinstance(dt, datetime_tz):
    if not dt.tzinfo:
      return datetime_tz(dt, tzinfo=localtz())
    dt = datetime_tz(dt)
  if force_to_local:
    return dt.astimezone(localtz())
  return dt