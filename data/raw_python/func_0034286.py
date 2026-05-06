def protract(self, x):
		"""
		Execute the .protract() method on each segment in the list
		and coalesce the result.  Segmentlist is modified in place.
		"""
		for i in xrange(len(self)):
			self[i] = self[i].protract(x)
		return self.coalesce()