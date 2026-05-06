def updateKeyMapping(self, mapping):
		"""
		Used as the first half of the row key reassignment
		algorithm.  Accepts a dictionary mapping old key --> new
		key.  Iterates over the rows in this table, using the
		table's next_id attribute to assign a new ID to each row,
		recording the changes in the mapping.  Returns the mapping.
		Raises ValueError if the table's next_id attribute is None.
		"""
		if self.next_id is None:
			raise ValueError(self)
		try:
			column = self.getColumnByName(self.next_id.column_name)
		except KeyError:
			# table is missing its ID column, this is a no-op
			return mapping
		for i, old in enumerate(column):
			if old is None:
				raise ValueError("null row ID encountered in Table '%s', row %d" % (self.getAttribute("Name"), i))
			if old in mapping:
				column[i] = mapping[old]
			else:
				column[i] = mapping[old] = self.get_next_id()
		return mapping