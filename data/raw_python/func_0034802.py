def _short_ts_regexp():
	'''Generates regexp for parsing of
		shortened relative timestamps, as shown in the table.'''
	ts_re = ['^']
	for k in it.chain(_short_ts_days, _short_ts_s):
		ts_re.append(r'(?P<{0}>\d+{0}\s*)?'.format(k))
	return re.compile(''.join(ts_re), re.I | re.U)