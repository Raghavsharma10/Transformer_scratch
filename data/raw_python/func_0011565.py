def normalize_file(file, separators=None):
	"""
	Normalizes the file path to use the POSIX path separator (i.e., ``'/'``).

	*file* (:class:`str`) is the file path.

	*separators* (:class:`~collections.abc.Collection` of :class:`str`; or
	:data:`None`) optionally contains the path separators to normalize.
	This does not need to include the POSIX path separator (``'/'``), but
	including it will not affect the results. Default is :data:`None` for
	:data:`NORMALIZE_PATH_SEPS`. To prevent normalization, pass an empty
	container (e.g., an empty tuple ``()``).

	Returns the normalized file path (:class:`str`).
	"""
	# Normalize path separators.
	if separators is None:
		separators = NORMALIZE_PATH_SEPS
	norm_file = file
	for sep in separators:
		norm_file = norm_file.replace(sep, posixpath.sep)

	# Remove current directory prefix.
	if norm_file.startswith('./'):
		norm_file = norm_file[2:]

	return norm_file