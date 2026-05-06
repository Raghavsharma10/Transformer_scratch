def _tzinfome(tzinfo):
  """Gets a tzinfo object from a string.

  Args:
    tzinfo: A string (or string like) object, or a datetime.tzinfo object.

  Returns:
    An datetime.tzinfo object.

  Raises:
    UnknownTimeZoneError: If the timezone given can't be decoded.
  """
  if not isinstance(tzinfo, datetime.tzinfo):
    try:
      tzinfo = pytz.timezone(tzinfo)
      assert tzinfo.zone in pytz.all_timezones
    except AttributeError:
      raise pytz.UnknownTimeZoneError("Unknown timezone! %s" % tzinfo)
  return tzinfo