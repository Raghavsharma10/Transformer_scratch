def get_column(self, column):
		"""@returns: an array of column values for each row in the table

		@param column:
			name of column to return
		@returntype:
			numpy.ndarray
		"""
		if column.lower() == 'q':
			return self.get_q
		else:
			return self.getColumnByName(column).asarray()