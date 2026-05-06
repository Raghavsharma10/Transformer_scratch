def get_coinc_def_id(self, search, search_coinc_type, create_new = True, description = None):
		"""
		Return the coinc_def_id for the row in the table whose
		search string and search_coinc_type integer have the values
		given.  If a matching row is not found, the default
		behaviour is to create a new row and return the ID assigned
		to the new row.  If, instead, create_new is False then
		KeyError is raised when a matching row is not found.  The
		optional description parameter can be used to set the
		description string assigned to the new row if one is
		created, otherwise the new row is left with no description.
		"""
		# look for the ID
		rows = [row for row in self if (row.search, row.search_coinc_type) == (search, search_coinc_type)]
		if len(rows) > 1:
			raise ValueError("(search, search coincidence type) = ('%s', %d) is not unique" % (search, search_coinc_type))
		if len(rows) > 0:
			return rows[0].coinc_def_id

		# coinc type not found in table
		if not create_new:
			raise KeyError((search, search_coinc_type))
		row = self.RowType()
		row.coinc_def_id = self.get_next_id()
		row.search = search
		row.search_coinc_type = search_coinc_type
		row.description = description
		self.append(row)

		# return new ID
		return row.coinc_def_id