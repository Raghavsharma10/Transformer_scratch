def match_file(self, file, separators=None):
		"""
		Matches the file to this path-spec.

		*file* (:class:`str`) is the file path to be matched against
		:attr:`self.patterns <PathSpec.patterns>`.

		*separators* (:class:`~collections.abc.Collection` of :class:`str`)
		optionally contains the path separators to normalize. See
		:func:`~pathspec.util.normalize_file` for more information.

		Returns :data:`True` if *file* matched; otherwise, :data:`False`.
		"""
		norm_file = util.normalize_file(file, separators=separators)
		return util.match_file(self.patterns, norm_file)