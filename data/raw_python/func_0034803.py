def parse_timestamp(ts_str):
	'''Match time either in human-readable format (as accepted by dateutil),
		or same time-offset format, as used in the table (e.g. "NdMh ago", or just "NdMh").'''
	assert isinstance(ts_str, bytes), [type(ts_str), repr(ts_str)]
	ts_str = ts_str.replace('_', ' ')

	# Try to parse time offset in short format
	match = _short_ts_regexp.search(ts_str)
	if match and any(match.groups()):
		delta = list()
		parse_int = lambda v: int(''.join(c for c in v if c.isdigit()))
		for units in [_short_ts_days, _short_ts_s]:
			val = 0
			for k, v in units.iteritems():
				try:
					if not match.group(k): continue
					n = parse_int(match.group(k))
				except IndexError: continue
				val += n * v
			delta.append(val)
		return timezone.localtime(timezone.now()) - timedelta(*delta)

	# Fallback to other generic formats
	ts = None
	if not ts:
		match = re.search( # common BE format
			r'^(?P<date>(?:\d{2}|(?P<Y>\d{4}))-\d{2}-\d{2})'
			r'(?:[ T](?P<time>\d{2}(?::\d{2}(?::\d{2})?)?)?)?$', ts_str )
		if match:
			tpl = 'y' if not match.group('Y') else 'Y'
			tpl, ts_str = '%{}-%m-%d'.format(tpl), match.group('date')
			if match.group('time'):
				tpl_time = ['%H', '%M', '%S']
				ts_str_time = match.group('time').split(':')
				ts_str += ' ' + ':'.join(ts_str_time)
				tpl += ' ' + ':'.join(tpl_time[:len(ts_str_time)])
			try: ts = timezone.make_aware(datetime.strptime(ts_str, tpl))
			except ValueError: pass
	if not ts:
		# coreutils' "date" parses virtually everything, but is more expensive to use
		with open(os.devnull, 'w') as devnull:
			proc = subprocess.Popen(
				['date', '+%s', '-d', ts_str],
				stdout=subprocess.PIPE, stderr=devnull, close_fds=True )
			val = proc.stdout.read()
			if not proc.wait():
				ts = timezone.make_aware(datetime.fromtimestamp(int(val.strip())))

	if ts: return ts
	raise ValueError('Unable to parse date/time string: {0}'.format(ts_str))