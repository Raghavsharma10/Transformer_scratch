def get_in_segmentlistdict(self, process_ids = None):
		"""
		Return a segmentlistdict mapping instrument to in segment
		list.  If process_ids is a sequence of process IDs, then
		only rows with matching IDs are included otherwise all rows
		are included.

		Note:  the result is not coalesced, each segmentlist
		contains the segments listed for that instrument as they
		appeared in the table.
		"""
		seglists = segments.segmentlistdict()
		for row in self:
			ifos = row.instruments or (None,)
			if process_ids is None or row.process_id in process_ids:
				seglists.extend(dict((ifo, segments.segmentlist([row.in_segment])) for ifo in ifos))
		return seglists