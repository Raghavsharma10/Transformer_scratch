def _date_to_datetime(value):
  """Convert a date to a datetime for Cloud Datastore storage.

  Args:
    value: A datetime.date object.

  Returns:
    A datetime object with time set to 0:00.
  """
  if not isinstance(value, datetime.date):
    raise TypeError('Cannot convert to datetime expected date value; '
                    'received %s' % value)
  return datetime.datetime(value.year, value.month, value.day)