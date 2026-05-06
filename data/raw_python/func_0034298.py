def extend(self, other):
		"""
		Appends the segmentlists from other to the corresponding
		segmentlists in self, adding new segmentslists to self as
		needed.
		"""
		for key, value in other.iteritems():
			if key not in self:
				self[key] = _shallowcopy(value)
			else:
				self[key].extend(value)