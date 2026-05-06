def getTablesByName(elem, name):
	"""
	Return a list of Table elements named name under elem.  The name
	comparison is done using CompareTableNames().
	"""
	name = StripTableName(name)
	return elem.getElements(lambda e: (e.tagName == ligolw.Table.tagName) and (e.Name == name))