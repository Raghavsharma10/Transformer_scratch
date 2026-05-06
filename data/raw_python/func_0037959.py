def _get_rev_num(self, rev=None):
		"""
		Determine the revision number for a given revision specifier.
		"""
		# first, determine the numeric ID
		cmd = ['identify', '--num']
		# workaround for #4
		cmd.extend(['--config', 'defaults.identify='])
		if rev:
			cmd.extend(['--rev', rev])
		res = self._invoke(*cmd)
		return res.strip()