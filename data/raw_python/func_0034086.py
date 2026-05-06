def sync_next_id(self):
		"""
		Determines the highest-numbered ID in this table, and sets
		the table's .next_id attribute to the next highest ID in
		sequence.  If the .next_id attribute is already set to a
		value greater than the highest value found, then it is left
		unmodified.  The return value is the ID identified by this
		method.  If the table's .next_id attribute is None, then
		this function is a no-op.

		Note that tables of the same name typically share a common
		.next_id attribute (it is a class attribute, not an
		attribute of each instance) so that IDs can be generated
		that are unique across all tables in the document.  Running
		sync_next_id() on all the tables in a document that are of
		the same type will have the effect of setting the ID to the
		next ID higher than any ID in any of those tables.

		Example:

		>>> import lsctables
		>>> tbl = lsctables.New(lsctables.ProcessTable)
		>>> print tbl.sync_next_id()
		process:process_id:0
		"""
		if self.next_id is not None:
			if len(self):
				n = max(self.getColumnByName(self.next_id.column_name)) + 1
			else:
				n = type(self.next_id)(0)
			if n > self.next_id:
				self.set_next_id(n)
		return self.next_id