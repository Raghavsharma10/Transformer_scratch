def getParamsByName(elem, name):
	"""
	Return a list of params with name name under elem.
	"""
	name = StripParamName(name)
	return elem.getElements(lambda e: (e.tagName == ligolw.Param.tagName) and (e.Name == name))