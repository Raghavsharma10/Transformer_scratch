def contract(self, x):
		"""
		Run .contract(x) on all segmentlists.
		"""
		for value in self.itervalues():
			value.contract(x)
		return self