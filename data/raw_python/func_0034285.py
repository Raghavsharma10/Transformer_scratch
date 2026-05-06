def coalesce(self):
		"""
		Sort the elements of the list into ascending order, and merge
		continuous segments into single segments.  Segmentlist is
		modified in place.  This operation is O(n log n).
		"""
		self.sort()
		i = j = 0
		n = len(self)
		while j < n:
			lo, hi = self[j]
			j += 1
			while j < n and hi >= self[j][0]:
				hi = max(hi, self[j][1])
				j += 1
			if lo != hi:
				self[i] = segment(lo, hi)
				i += 1
		del self[i : ]
		return self