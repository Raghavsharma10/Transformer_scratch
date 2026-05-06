def match_files(patterns, files):
	"""
	Matches the files to the patterns.

	*patterns* (:class:`~collections.abc.Iterable` of :class:`~pathspec.pattern.Pattern`)
	contains the patterns to use.

	*files* (:class:`~collections.abc.Iterable` of :class:`str`) contains
	the normalized file paths to be matched against *patterns*.

	Returns the matched files (:class:`set` of :class:`str`).
	"""
	all_files = files if isinstance(files, collection_type) else list(files)
	return_files = set()
	for pattern in patterns:
		if pattern.include is not None:
			result_files = pattern.match(all_files)
			if pattern.include:
				return_files.update(result_files)
			else:
				return_files.difference_update(result_files)
	return return_files