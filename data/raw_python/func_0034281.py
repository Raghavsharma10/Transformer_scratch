def extent(self):
		"""
		Return the segment whose end-points denote the maximum and
		minimum extent of the segmentlist.  Does not require the
		segmentlist to be coalesced.
		"""
		if not len(self):
			raise ValueError("empty list")
		min, max = self[0]
		for lo, hi in self:
			if min > lo:
				min = lo
			if max < hi:
				max = hi
		return segment(min, max)