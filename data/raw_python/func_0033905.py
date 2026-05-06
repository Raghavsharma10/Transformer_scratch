def compare_table_cols(a, b):
	"""
	Return False if the two tables a and b have the same columns
	(ignoring order) according to LIGO LW name conventions, return True
	otherwise.
	"""
	return cmp(sorted((col.Name, col.Type) for col in a.getElementsByTagName(ligolw.Column.tagName)), sorted((col.Name, col.Type) for col in b.getElementsByTagName(ligolw.Column.tagName)))