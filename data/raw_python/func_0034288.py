def shift(self, x):
		"""
		Execute the .shift() method on each segment in the list.
		The algorithm is O(n) and does not require the list to be
		coalesced nor does it coalesce the list.  Segmentlist is
		modified in place.
		"""
		for i in xrange(len(self)):
			self[i] = self[i].shift(x)
		return self