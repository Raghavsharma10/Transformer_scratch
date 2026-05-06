def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)