def _read_tags_for_revset(self, spec):
		"""
		Return TaggedRevision for each tag/rev combination in the revset spec
		"""
		cmd = [
			'log', '--style', 'default', '--config', 'defaults.log=',
			'-r', spec]
		res = self._invoke(*cmd)
		header_pattern = re.compile(r'(?P<header>\w+?):\s+(?P<value>.*)')
		match_res = map(header_pattern.match, res.splitlines())
		matched_lines = filter(None, match_res)
		matches = (match.groupdict() for match in matched_lines)
		for match in matches:
			if match['header'] == 'changeset':
				id, sep, rev = match['value'].partition(':')
			if match['header'] == 'tag':
				tag = match['value']
				yield TaggedRevision(tag, rev)