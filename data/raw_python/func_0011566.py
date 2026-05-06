def normalize_files(files, separators=None):
	"""
	Normalizes the file paths to use the POSIX path separator.

	*files* (:class:`~collections.abc.Iterable` of :class:`str`) contains
	the file paths to be normalized.

	*separators* (:class:`~collections.abc.Collection` of :class:`str`; or
	:data:`None`) optionally contains the path separators to normalize.
	See :func:`normalize_file` for more information.

	Returns a :class:`dict` mapping the each normalized file path (:class:`str`)
	to the original file path (:class:`str`)
	"""
	norm_files = {}
	for path in files:
		norm_files[normalize_file(path, separators=separators)] = path
	return norm_files