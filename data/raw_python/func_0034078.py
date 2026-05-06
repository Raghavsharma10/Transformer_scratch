def copy(self):
		"""
		Construct and return a new Table document subtree whose
		structure is the same as this table, that is it has the
		same columns etc..  The rows are not copied.  Note that a
		fair amount of metadata is shared between the original and
		new tables.  In particular, a copy of the Table object
		itself is created (but with no rows), and copies of the
		child nodes are created.  All other object references are
		shared between the two instances, such as the RowType
		attribute on the Table object.
		"""
		new = copy.copy(self)
		new.childNodes = map(copy.copy, self.childNodes)
		for child in new.childNodes:
			child.parentNode = new
		del new[:]
		new._end_of_columns()
		new._end_of_rows()
		return new