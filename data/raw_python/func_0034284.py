def intersects(self, other):
		"""
		Returns True if the intersection of self and the
		segmentlist other is not the null set, otherwise returns
		False.  The algorithm is O(n), but faster than explicit
		calculation of the intersection, i.e. by testing bool(self
		& other).  Requires both lists to be coalesced.
		"""
		# if either has zero length, the answer is False
		if not (self and other):
			return False
		# walk through both lists in order, searching for a match
		i = j = 0
		seg = self[0]
		otherseg = other[0]
		while True:
			if seg[1] <= otherseg[0]:
				i += 1
				if i >= len(self):
					return False
				seg = self[i]
			elif otherseg[1] <= seg[0]:
				j += 1
				if j >= len(other):
					return False
				otherseg = other[j]
			else:
				return True