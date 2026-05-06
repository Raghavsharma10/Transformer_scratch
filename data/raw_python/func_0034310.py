def segmentlistdict_fromsearchsummary_in(xmldoc, program = None):
	"""
	Convenience wrapper for a common case usage of the segmentlistdict
	class:  searches the process table in xmldoc for occurances of a
	program named program, then scans the search summary table for
	matching process IDs and constructs a segmentlistdict object from
	the in segments in those rows.

	Note:  the segmentlists in the segmentlistdict are not necessarily
	coalesced, they contain the segments as they appear in the
	search_summary table.
	"""
	stbl = lsctables.SearchSummaryTable.get_table(xmldoc)
	ptbl = lsctables.ProcessTable.get_table(xmldoc)
	return stbl.get_in_segmentlistdict(program and ptbl.get_ids_by_program(program))