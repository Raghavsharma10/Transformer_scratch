def row_from_cols(self, values):
		"""
		Given an iterable of values in the order of columns in the
		database, construct and return a row object.  This is a
		convenience function for turning the results of database
		queries into Python objects.
		"""
		row = self.RowType()
		for c, t, v in zip(self.dbcolumnnames, self.dbcolumntypes, values):
			if t in ligolwtypes.IDTypes:
				v = ilwd.ilwdchar(v)
			setattr(row, c, v)
		return row