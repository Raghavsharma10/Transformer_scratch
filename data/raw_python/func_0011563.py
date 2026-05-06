def match_file(patterns, file):
	"""
	Matches the file to the patterns.

	*patterns* (:class:`~collections.abc.Iterable` of :class:`~pathspec.pattern.Pattern`)
	contains the patterns to use.

	*file* (:class:`str`) is the normalized file path to be matched
	against *patterns*.

	Returns :data:`True` if *file* matched; otherwise, :data:`False`.
	"""
	matched = False
	for pattern in patterns:
		if pattern.include is not None:
			if file in pattern.match((file,)):
				matched = pattern.include
	return matched