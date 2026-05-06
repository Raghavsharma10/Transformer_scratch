def merge_compatible_tables(elem):
	"""
	Below the given element, find all Tables whose structure is
	described in lsctables, and merge compatible ones of like type.
	That is, merge all SnglBurstTables that have the same columns into
	a single table, etc..
	"""
	for name in lsctables.TableByName.keys():
		tables = table.getTablesByName(elem, name)
		if tables:
			dest = tables.pop(0)
			for src in tables:
				if src.Name != dest.Name:
					# src and dest have different names
					continue
				# src and dest have the same names
				if compare_table_cols(dest, src):
					# but they have different columns
					raise ValueError("document contains %s tables with incompatible columns" % dest.Name)
				# and the have the same columns
				# copy src rows to dest
				for row in src:
					dest.append(row)
				# unlink src from parent
				if src.parentNode is not None:
					src.parentNode.removeChild(src)
	return elem