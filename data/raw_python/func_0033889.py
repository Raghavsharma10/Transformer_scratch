def getArraysByName(elem, name):
	"""
	Return a list of arrays with name name under elem.
	"""
	name = StripArrayName(name)
	return elem.getElements(lambda e: (e.tagName == ligolw.Array.tagName) and (e.Name == name))