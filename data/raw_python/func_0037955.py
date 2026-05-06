def version(self):
		"""
		Return the underlying version
		"""
		lines = iter(self._invoke('version').splitlines())
		version = next(lines).strip()
		return self._parse_version(version)