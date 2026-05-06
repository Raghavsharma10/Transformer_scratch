def update(self, d):
		"""
		From a dictionary of offsets, apply each offset to the
		corresponding segmentlist.  NOTE:  it is acceptable for the
		offset dictionary to contain entries for which there is no
		matching segmentlist; no error will be raised, but the
		offset will be ignored.  This simplifies the case of
		updating several segmentlistdict objects from a common
		offset dictionary, when one or more of the segmentlistdicts
		contains only a subset of the keys.
		"""
		for key, value in d.iteritems():
			if key in self:
				self[key] = value