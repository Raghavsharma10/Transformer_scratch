def getColumnByName(self, name):
		"""
		Retrieve and return the Column child element named name.
		The comparison is done using CompareColumnNames().  Raises
		KeyError if this table has no column by that name.

		Example:

		>>> import lsctables
		>>> tbl = lsctables.New(lsctables.SnglInspiralTable)
		>>> col = tbl.getColumnByName("mass1")
		"""
		try:
			col, = getColumnsByName(self, name)
		except ValueError:
			# did not find exactly 1 matching child
			raise KeyError(name)
		return col