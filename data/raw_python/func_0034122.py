def process_params_from_dict(paramdict):
	"""
	Generator function yields (name, type, value) tuples constructed
	from a dictionary of name/value pairs.  The tuples are suitable for
	input to append_process_params().  This is intended as a
	convenience for converting command-line options into process_params
	rows.  The name values in the output have "--" prepended to them
	and all "_" characters replaced with "-".  The type strings are
	guessed from the Python types of the values.  If a value is a
	Python list (or instance of a subclass thereof), then one tuple is
	produced for each of the items in the list.

	Example:

	>>> list(process_params_from_dict({"verbose": True, "window": 4.0, "include": ["/tmp", "/var/tmp"]}))
	[(u'--window', u'real_8', 4.0), (u'--verbose', None, None), (u'--include', u'lstring', '/tmp'), (u'--include', u'lstring', '/var/tmp')]
	"""
	for name, values in paramdict.items():
		# change the name back to the form it had on the command line
		name = u"--%s" % name.replace("_", "-")

		if values is True or values is False:
			yield (name, None, None)
		elif values is not None:
			if not isinstance(values, list):
				values = [values]
			for value in values:
				yield (name, ligolwtypes.FromPyType[type(value)], value)