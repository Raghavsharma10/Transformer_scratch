def applyKeyMapping(self, mapping):
		"""
		Used as the second half of the key reassignment algorithm.
		Loops over each row in the table, replacing references to
		old row keys with the new values from the mapping.
		"""
		for coltype, colname in zip(self.columntypes, self.columnnames):
			if coltype in ligolwtypes.IDTypes and (self.next_id is None or colname != self.next_id.column_name):
				column = self.getColumnByName(colname)
				for i, old in enumerate(column):
					try:
						column[i] = mapping[old]
					except KeyError:
						pass