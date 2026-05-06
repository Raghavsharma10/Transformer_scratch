def protract(self, x):
		"""
		Run .protract(x) on all segmentlists.
		"""
		for value in self.itervalues():
			value.protract(x)
		return self