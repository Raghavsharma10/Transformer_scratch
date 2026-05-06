def get_param(xmldoc, name):
	"""
	Scan xmldoc for a param named name.  Raises ValueError if not
	exactly 1 such param is found.
	"""
	params = getParamsByName(xmldoc, name)
	if len(params) != 1:
		raise ValueError("document must contain exactly one %s param" % StripParamName(name))
	return params[0]