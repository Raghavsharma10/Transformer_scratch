def index(self, value):
		"""
		Return the smallest index of the row(s) with this column
		equal to value.
		"""
		for i in xrange(len(self.parentNode)):
			if getattr(self.parentNode[i], self.Name) == value:
				return i
		raise ValueError(value)