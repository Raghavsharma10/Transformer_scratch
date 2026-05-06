def to_segmentlistdict(self):
		"""
		Return a segmentlistdict object describing the instruments
		and times spanned by the entries in this Cache.  The return
		value is coalesced.
		"""
		d = segments.segmentlistdict()
		for entry in self:
			d |= entry.segmentlistdict
		return d