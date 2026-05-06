def contract(self, x):
		"""
		Execute the .contract() method on each segment in the list
		and coalesce the result.  Segmentlist is modified in place.
		"""
		for i in xrange(len(self)):
			self[i] = self[i].contract(x)
		return self.coalesce()