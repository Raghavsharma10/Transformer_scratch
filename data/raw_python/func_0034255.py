def get_by_name(self, name, clip_to_valid = False):
		"""
		Retrieve the active segmentlists whose name equals name.
		The result is a segmentlistdict indexed by instrument.  All
		segmentlist objects within it will be copies of the
		contents of this object, modifications will not affect the
		contents of this object.  If clip_to_valid is True then the
		segmentlists will be intersected with their respective
		intervals of validity, otherwise they will be the verbatim
		active segments.

		NOTE:  the intersection operation required by clip_to_valid
		will yield undefined results unless the active and valid
		segmentlist objects are coalesced.
		"""
		result = segments.segmentlistdict()
		for seglist in self:
			if seglist.name != name:
				continue
			segs = seglist.active
			if clip_to_valid:
				# do not use in-place intersection
				segs = segs & seglist.valid
			for instrument in seglist.instruments:
				if instrument in result:
					raise ValueError("multiple '%s' segmentlists for instrument '%s'" % (name, instrument))
				result[instrument] = segs.copy()
		return result