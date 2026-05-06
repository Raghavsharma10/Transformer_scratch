def get(self, segment_def_id = None):
		"""
		Return a segmentlist object describing the times spanned by
		the segments carrying the given segment_def_id.  If
		segment_def_id is None then all segments are returned.

		Note:  the result is not coalesced, the segmentlist
		contains the segments as they appear in the table.
		"""
		if segment_def_id is None:
			return segments.segmentlist(row.segment for row in self)
		return segments.segmentlist(row.segment for row in self if row.segment_def_id == segment_def_id)