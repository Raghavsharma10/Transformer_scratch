def _time_to_datetime(value):
  """Convert a time to a datetime for Cloud Datastore storage.

  Args:
    value: A datetime.time object.

  Returns:
    A datetime object with date set to 1970-01-01.
  """
  if not isinstance(value, datetime.time):
    raise TypeError('Cannot convert to datetime expected time value; '
                    'received %s' % value)
  return datetime.datetime(1970, 1, 1,
                           value.hour, value.minute, value.second,
                           value.microsecond)