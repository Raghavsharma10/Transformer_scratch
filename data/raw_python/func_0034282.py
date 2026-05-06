def find(self, item):
		"""
		Return the smallest i such that i is the index of an
		element that wholly contains item.  Raises ValueError if no
		such element exists.  Does not require the segmentlist to
		be coalesced.
		"""
		for i, seg in enumerate(self):
			if item in seg:
				return i
		raise ValueError(item)