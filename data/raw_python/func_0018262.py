def _CopyTimeFromString(self, time_string):
    """Copies a time from a string.

    Args:
      time_string (str): time value formatted as:
          hh:mm:ss.######[+-]##:##

          Where # are numeric digits ranging from 0 to 9 and the seconds
          fraction can be either 3 or 6 digits. The seconds fraction and
          time zone offset are optional.

    Returns:
      tuple[int, int, int, int, int]: hours, minutes, seconds, microseconds,
          time zone offset in minutes.

    Raises:
      ValueError: if the time string is invalid or not supported.
    """
    time_string_length = len(time_string)

    # The time string should at least contain 'hh:mm:ss'.
    if time_string_length < 8:
      raise ValueError('Time string too short.')

    if time_string[2] != ':' or time_string[5] != ':':
      raise ValueError('Invalid time string.')

    try:
      hours = int(time_string[0:2], 10)
    except ValueError:
      raise ValueError('Unable to parse hours.')

    if hours not in range(0, 24):
      raise ValueError('Hours value: {0:d} out of bounds.'.format(hours))

    try:
      minutes = int(time_string[3:5], 10)
    except ValueError:
      raise ValueError('Unable to parse minutes.')

    if minutes not in range(0, 60):
      raise ValueError('Minutes value: {0:d} out of bounds.'.format(minutes))

    try:
      seconds = int(time_string[6:8], 10)
    except ValueError:
      raise ValueError('Unable to parse day of seconds.')

    # TODO: support a leap second?
    if seconds not in range(0, 60):
      raise ValueError('Seconds value: {0:d} out of bounds.'.format(seconds))

    microseconds = None
    time_zone_offset = None

    time_zone_string_index = 8
    while time_zone_string_index < time_string_length:
      if time_string[time_zone_string_index] in ('+', '-'):
        break

      time_zone_string_index += 1

    # The calculations that follow rely on the time zone string index
    # to point beyond the string in case no time zone offset was defined.
    if time_zone_string_index == time_string_length - 1:
      time_zone_string_index += 1

    if time_string_length > 8 and time_string[8] == '.':
      time_fraction_length = time_zone_string_index - 9
      if time_fraction_length not in (3, 6):
        raise ValueError('Invalid time string.')

      try:
        time_fraction = time_string[9:time_zone_string_index]
        time_fraction = int(time_fraction, 10)
      except ValueError:
        raise ValueError('Unable to parse time fraction.')

      if time_fraction_length == 3:
        time_fraction *= 1000

      microseconds = time_fraction

    if time_zone_string_index < time_string_length:
      if (time_string_length - time_zone_string_index != 6 or
          time_string[time_zone_string_index + 3] != ':'):
        raise ValueError('Invalid time string.')

      try:
        hours_from_utc = int(time_string[
            time_zone_string_index + 1:time_zone_string_index + 3])
      except ValueError:
        raise ValueError('Unable to parse time zone hours offset.')

      if hours_from_utc not in range(0, 15):
        raise ValueError('Time zone hours offset value out of bounds.')

      try:
        minutes_from_utc = int(time_string[
            time_zone_string_index + 4:time_zone_string_index + 6])
      except ValueError:
        raise ValueError('Unable to parse time zone minutes offset.')

      if minutes_from_utc not in range(0, 60):
        raise ValueError('Time zone minutes offset value out of bounds.')

      # pylint: disable=invalid-unary-operand-type
      time_zone_offset = (hours_from_utc * 60) + minutes_from_utc

      # Note that when the sign of the time zone offset is negative
      # the difference needs to be added. We do so by flipping the sign.
      if time_string[time_zone_string_index] != '-':
        time_zone_offset = -time_zone_offset

    return hours, minutes, seconds, microseconds, time_zone_offset