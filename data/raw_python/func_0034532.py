def get_modified_date(parsed, raw):
	'Return best possible guess to post modification timestamp.'
	if parsed: return feedparser_ts(parsed)
	if not raw: return None

	# Parse weird timestamps that feedparser can't handle, e.g.: July 30, 2013
	ts, val = None, raw.replace('_', ' ')
	if not ts:
		# coreutils' "date" parses virtually everything, but is more expensive to use
		from subprocess import Popen, PIPE
		with open(os.devnull, 'w') as devnull:
			proc = Popen(['date', '+%s', '-d', val], stdout=PIPE, stderr=devnull)
			val = proc.stdout.read()
			if not proc.wait():
				ts = datetime.fromtimestamp(int(val.strip()), tz=timezone.utc)
	if ts: return ts
	raise ValueError('Unrecognized raw value format: {0!r}'.format(val))