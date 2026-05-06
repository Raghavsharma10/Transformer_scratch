def extract_common(self, keys):
		"""
		Return a new segmentlistdict containing only those
		segmentlists associated with the keys in keys, with each
		set to their mutual intersection.  The offsets are
		preserved.
		"""
		keys = set(keys)
		new = self.__class__()
		intersection = self.intersection(keys)
		for key in keys:
			dict.__setitem__(new, key, _shallowcopy(intersection))
			dict.__setitem__(new.offsets, key, self.offsets[key])
		return new