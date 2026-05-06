def intersection(self, keys):
		"""
		Return the intersection of the segmentlists associated with
		the keys in keys.
		"""
		keys = set(keys)
		if not keys:
			return segmentlist()
		seglist = _shallowcopy(self[keys.pop()])
		for key in keys:
			seglist &= self[key]
		return seglist