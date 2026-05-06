def unix_timestamp_to_datetime(unix_timestamp):
  """
  <Purpose>
    Convert 'unix_timestamp' (i.e., POSIX time, in UNIX_TIMESTAMP_SCHEMA format)
    to a datetime.datetime() object.  'unix_timestamp' is the number of seconds
    since the epoch (January 1, 1970.)

    >>> datetime_object = unix_timestamp_to_datetime(1445455680)
    >>> datetime_object
    datetime.datetime(2015, 10, 21, 19, 28)

  <Arguments>
    unix_timestamp:
      An integer representing the time (e.g., 1445455680).  Conformant to
      'securesystemslib.formats.UNIX_TIMESTAMP_SCHEMA'.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'unix_timestamp' is improperly
    formatted.

  <Side Effects>
    None.

  <Returns>
    A datetime.datetime() object corresponding to 'unix_timestamp'.
  """

  # Is 'unix_timestamp' properly formatted?
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.UNIX_TIMESTAMP_SCHEMA.check_match(unix_timestamp)

  # Convert 'unix_timestamp' to a 'time.struct_time',  in UTC.  The Daylight
  # Savings Time (DST) flag is set to zero.  datetime.fromtimestamp() is not
  # used because it returns a local datetime.
  struct_time = time.gmtime(unix_timestamp)

  # Extract the (year, month, day, hour, minutes, seconds) arguments for the
  # datetime object to be returned.
  datetime_object = datetime.datetime(*struct_time[:6])

  return datetime_object