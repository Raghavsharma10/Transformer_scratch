def get_array(xmldoc, name):
	"""
	Scan xmldoc for an array named name.  Raises ValueError if not
	exactly 1 such array is found.
	"""
	arrays = getArraysByName(xmldoc, name)
	if len(arrays) != 1:
		raise ValueError("document must contain exactly one %s array" % StripArrayName(name))
	return arrays[0]