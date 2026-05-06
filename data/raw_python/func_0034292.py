def extent_all(self):
		"""
		Return the result of running .extent() on the union of all
		lists in the dictionary.
		"""
		segs = tuple(seglist.extent() for seglist in self.values() if seglist)
		if not segs:
			raise ValueError("empty list")
		return segment(min(seg[0] for seg in segs), max(seg[1] for seg in segs))