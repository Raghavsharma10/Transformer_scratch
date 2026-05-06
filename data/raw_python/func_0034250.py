def segmenttable_get_by_name(xmldoc, name):
	"""
	Retrieve the segmentlists whose name equals name.  The result is a
	segmentlistdict indexed by instrument.

	The output of this function is not coalesced, each segmentlist
	contains the segments as found in the segment table.

	NOTE:  this is a light-weight version of the .get_by_name() method
	of the LigolwSegments class intended for use when the full
	machinery of that class is not required.  Considerably less
	document validation and error checking is performed by this
	version.  Consider using that method instead if your application
	will be interfacing with the document via that class anyway.
	"""
	#
	# find required tables
	#

	def_table = lsctables.SegmentDefTable.get_table(xmldoc)
	seg_table = lsctables.SegmentTable.get_table(xmldoc)

	#
	# segment_def_id --> instrument names mapping but only for
	# segment_definer entries bearing the requested name
	#

	instrument_index = dict((row.segment_def_id, row.instruments) for row in def_table if row.name == name)

	#
	# populate result segmentlistdict object from segment_def_map table
	# and index
	#

	instruments = set(instrument for instruments in instrument_index.values() for instrument in instruments)
	result = segments.segmentlistdict((instrument, segments.segmentlist()) for instrument in instruments)

	for row in seg_table:
		if row.segment_def_id in instrument_index:
			seg = row.segment
			for instrument in instrument_index[row.segment_def_id]:
				result[instrument].append(seg)

	#
	# done
	#

	return result