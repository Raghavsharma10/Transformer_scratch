def CopyFromStringISO8601(self, time_string):
    """Copies time elements from an ISO 8601 date and time string.

    Currently not supported:
    * Duration notation: "P..."
    * Week notation "2016-W33"
    * Date with week number notation "2016-W33-3"
    * Date without year notation "--08-17"
    * Ordinal date notation "2016-230"

    Args:
      time_string (str): date and time value formatted as:
          YYYY-MM-DDThh:mm:ss.######[+-]##:##

          Where # are numeric digits ranging from 0 to 9 and the seconds
          fraction can be either 3 or 6 digits. The time of day, seconds
          fraction and time zone offset are optional. The default time zone
          is UTC.

    Raises:
      ValueError: if the time string is invalid or not supported.
    """
    date_time_values = self._CopyDateTimeFromStringISO8601(time_string)

    self._CopyFromDateTimeValues(date_time_values)