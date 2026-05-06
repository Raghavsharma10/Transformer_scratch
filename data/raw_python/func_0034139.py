def _append(self, row):
		"""
		Standard .append() method.  This method is for intended for
		internal use only.
		"""
		self.cursor.execute(self.append_statement, self.append_attrgetter(row))