def _date_trunc(value, timeframe):
  """
  A date flooring function.

  Returns the closest datetime to the current one that aligns to timeframe.
  For example, _date_trunc('2014-08-13 05:00:00', DateTrunc.Unit.MONTH)
  will return a Kronos time representing 2014-08-01 00:00:00.
  """
  if isinstance(value, types.StringTypes):
    value = parse(value)
    return_as_str = True
  else:
    value = kronos_time_to_datetime(value)
    return_as_str = False
  timeframes = {
    DateTrunc.Unit.SECOND: (lambda dt:
                            dt - timedelta(microseconds=dt.microsecond)),
    DateTrunc.Unit.MINUTE: (lambda dt:
                            dt - timedelta(seconds=dt.second,
                                           microseconds=dt.microsecond)),
    DateTrunc.Unit.HOUR: (lambda dt:
                          dt - timedelta(minutes=dt.minute,
                                         seconds=dt.second,
                                         microseconds=dt.microsecond)),
    DateTrunc.Unit.DAY: lambda dt: dt.date(),
    DateTrunc.Unit.WEEK: lambda dt: dt.date() - timedelta(days=dt.weekday()),
    DateTrunc.Unit.MONTH: lambda dt: datetime(dt.year, dt.month, 1),
    DateTrunc.Unit.YEAR: lambda dt: datetime(dt.year, 1, 1)
  }
  value = timeframes[timeframe](value)
  if return_as_str:
    return value.isoformat()
  return datetime_to_kronos_time(value)