def getColumnsByName(elem, name):
	"""
	Return a list of Column elements named name under elem.  The name
	comparison is done with CompareColumnNames().
	"""
	name = StripColumnName(name)
	return elem.getElements(lambda e: (e.tagName == ligolw.Column.tagName) and (e.Name == name))