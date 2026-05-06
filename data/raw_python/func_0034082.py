def appendRow(self, *args, **kwargs):
		"""
		Create and append a new row to this table, then return it

		All positional and keyword arguments are passed to the RowType
		constructor for this table.
		"""
		row = self.RowType(*args, **kwargs)
		self.append(row)
		return row