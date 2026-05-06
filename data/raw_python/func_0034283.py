def intersects_segment(self, other):
		"""
		Returns True if the intersection of self and the segment
		other is not the null set, otherwise returns False.  The
		algorithm is O(log n).  Requires the list to be coalesced.
		"""
		i = _bisect_left(self, other)
		return ((i != 0) and (other[0] < self[i-1][1])) or ((i != len(self)) and (other[1] > self[i][0]))