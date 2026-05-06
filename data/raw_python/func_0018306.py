def _CopyTimeFromStringISO8601(self, time_string):
    """Copies a time from an ISO 8601 date and time string.

    Args:
      time_string (str): time value formatted as:
          hh:mm:ss.######[+-]##:##

          Where # are numeric digits ranging from 0 to 9 and the seconds
          fraction can be either 3 or 6 digits. The faction of second and
          time zone offset are optional.

    Returns:
      tuple[int, int, int, int, int]: hours, minutes, seconds, microseconds,
          time zone offset in minutes.

    Raises:
      ValueError: if the time string is invalid or not supported.
    """
    if time_string.endswith('Z'):
      time_string = time_string[:-1]

    time_string_length = len(time_string)

    # The time string should at least contain 'hh'.
    if time_string_length < 2:
      raise ValueError('Time string too short.')

    try:
      hours = int(time_string[0:2], 10)
    except ValueError:
      raise ValueError('Unable to parse hours.')

    if hours not in range(0, 24):
      raise ValueError('Hours value: {0:d} out of bounds.'.format(hours))

    minutes = None
    seconds = None
    microseconds = None
    time_zone_offset = None

    time_string_index = 2

    # Minutes are either specified as 'hhmm', 'hh:mm' or as a fractional part
    # 'hh[.,]###'.
    if (time_string_index + 1 < time_string_length and
        time_string[time_string_index] not in ('.', ',')):
      if time_string[time_string_index] == ':':
        time_string_index += 1

      if time_string_index + 2 > time_string_length:
        raise ValueError('Time string too short.')

      try:
        minutes = time_string[time_string_index:time_string_index + 2]
        minutes = int(minutes, 10)
      except ValueError:
        raise ValueError('Unable to parse minutes.')

      time_string_index += 2

    # Seconds are either specified as 'hhmmss', 'hh:mm:ss' or as a fractional
    # part 'hh:mm[.,]###' or 'hhmm[.,]###'.
    if (time_string_index + 1 < time_string_length and
        time_string[time_string_index] not in ('.', ',')):
      if time_string[time_string_index] == ':':
        time_string_index += 1

      if time_string_index + 2 > time_string_length:
        raise ValueError('Time string too short.')

      try:
        seconds = time_string[time_string_index:time_string_index + 2]
        seconds = int(seconds, 10)
      except ValueError:
        raise ValueError('Unable to parse day of seconds.')

      time_string_index += 2

    time_zone_string_index = time_string_index
    while time_zone_string_index < time_string_length:
      if time_string[time_zone_string_index] in ('+', '-'):
        break

      time_zone_string_index += 1

    # The calculations that follow rely on the time zone string index
    # to point beyond the string in case no time zone offset was defined.
    if time_zone_string_index == time_string_length - 1:
      time_zone_string_index += 1

    if (time_string_length > time_string_index and
        time_string[time_string_index] in ('.', ',')):
      time_string_index += 1
      time_fraction_length = time_zone_string_index - time_string_index

      try:
        time_fraction = time_string[time_string_index:time_zone_string_index]
        time_fraction = int(time_fraction, 10)
        time_fraction = (
            decimal.Decimal(time_fraction) /
            decimal.Decimal(10 ** time_fraction_length))
      except ValueError:
        raise ValueError('Unable to parse time fraction.')

      if minutes is None:
        time_fraction *= 60
        minutes = int(time_fraction)
        time_fraction -= minutes

      if seconds is None:
        time_fraction *= 60
        seconds = int(time_fraction)
        time_fraction -= seconds

      time_fraction *= definitions.MICROSECONDS_PER_SECOND
      microseconds = int(time_fraction)

    if minutes is not None and minutes not in range(0, 60):
      raise ValueError('Minutes value: {0:d} out of bounds.'.format(minutes))

    # TODO: support a leap second?
    if seconds is not None and seconds not in range(0, 60):
      raise ValueError('Seconds value: {0:d} out of bounds.'.format(seconds))

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