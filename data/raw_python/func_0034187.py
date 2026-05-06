def HasNonLSCTables(elem):
	"""
	Return True if the document tree below elem contains non-LSC
	tables, otherwise return False.
	"""
	return any(t.Name not in TableByName for t in elem.getElementsByTagName(ligolw.Table.tagName))