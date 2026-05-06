def match(self, files):
		"""
		Matches this pattern against the specified files.

		*files* (:class:`~collections.abc.Iterable` of :class:`str`)
		contains each file relative to the root directory (e.g., "relative/path/to/file").

		Returns an :class:`~collections.abc.Iterable` yielding each matched
		file path (:class:`str`).
		"""
		if self.include is not None:
			for path in files:
				if self.regex.match(path) is not None:
					yield path