def CopyToDateTimeStringISO8601(self):
    """Copies the date time value to an ISO 8601 date and time string.

    Returns:
      str: date and time value formatted as an ISO 8601 date and time string or
          None if the timestamp cannot be copied to a date and time string.
    """
    date_time_string = self.CopyToDateTimeString()
    if date_time_string:
      date_time_string = date_time_string.replace(' ', 'T')
      date_time_string = '{0:s}Z'.format(date_time_string)
    return date_time_string