def timestamp(dt):
	"""
	Return POSIX timestamp as float.

	>>> timestamp(datetime.datetime.now()) > 1494638812
	True

	>>> timestamp(datetime.datetime.now()) % 1 > 0
	True
	"""
	if dt.tzinfo is None:
		return time.mktime((
			dt.year, dt.month, dt.day,
			dt.hour, dt.minute, dt.second,
			-1, -1, -1)) + dt.microsecond / 1e6
	else:
		return (dt - _EPOCH).total_seconds()