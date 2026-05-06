def count(self, value):
		"""
		Return the number of rows with this column equal to value.
		"""
		return sum(getattr(row, self.Name) == value for row in self.parentNode)