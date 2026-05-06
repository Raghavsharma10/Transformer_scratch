def get_naive(dt):
  """Gets a naive datetime from a datetime.

  datetime_tz objects can't just have tzinfo replaced with None, you need to
  call asdatetime.

  Args:
    dt: datetime object.

  Returns:
    datetime object without any timezone information.
  """
  if not dt.tzinfo:
    return dt
  if hasattr(dt, "asdatetime"):
    return dt.asdatetime()
  return dt.replace(tzinfo=None)