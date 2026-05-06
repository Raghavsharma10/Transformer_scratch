def _date_part(value, part):
  """
  Returns a portion of a datetime.

  Returns the portion of a datetime represented by timeframe.
  For example, _date_part('2014-08-13 05:00:00', DatePart.Unit.WEEK_DAY)
  will return 2, for Wednesday.
  """
  if isinstance(value, types.StringTypes):
    value = parse(value)
  else:
    value = kronos_time_to_datetime(value)
  parts = {
    DatePart.Unit.SECOND: lambda dt: dt.second,
    DatePart.Unit.MINUTE: lambda dt: dt.minute,
    DatePart.Unit.HOUR: lambda dt: dt.hour,
    DatePart.Unit.DAY: lambda dt: dt.day,
    DatePart.Unit.MONTH: lambda dt: dt.month,
    DatePart.Unit.YEAR: lambda dt: dt.year,
    DatePart.Unit.WEEK_DAY: lambda dt: dt.weekday(),
  }
  result = parts[part](value)
  return result