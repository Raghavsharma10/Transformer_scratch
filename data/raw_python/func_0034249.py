def has_segment_tables(xmldoc, name = None):
	"""
	Return True if the document contains a complete set of segment
	tables.  Returns False otherwise.  If name is given and not None
	then the return value is True only if the document's segment
	tables, if present, contain a segment list by that name.
	"""
	try:
		names = lsctables.SegmentDefTable.get_table(xmldoc).getColumnByName("name")
		lsctables.SegmentTable.get_table(xmldoc)
		lsctables.SegmentSumTable.get_table(xmldoc)
	except (ValueError, KeyError):
		return False
	return name is None or name in names