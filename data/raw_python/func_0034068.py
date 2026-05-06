def StripTableName(name):
	"""
	Return the significant portion of a table name according to LIGO LW
	naming conventions.

	Example:

	>>> StripTableName("sngl_burst_group:sngl_burst:table")
	'sngl_burst'
	>>> StripTableName("sngl_burst:table")
	'sngl_burst'
	>>> StripTableName("sngl_burst")
	'sngl_burst'
	"""
	if name.lower() != name:
		warnings.warn("table name \"%s\" is not lower case" % name)
	try:
		return TablePattern.search(name).group("Name")
	except AttributeError:
		return name