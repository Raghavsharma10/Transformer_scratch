def insert_from_segmentlistdict(self, seglists, name, version = None, comment = None, valid=None):
		"""
		Insert the segments from the segmentlistdict object
		seglists as a new list of "active" segments into this
		LigolwSegments object.  The dictionary's keys are assumed
		to provide the instrument name for each segment list.  A
		new entry will be created in the segment_definer table for
		the segment lists, and the dictionary's keys, the name, and
		comment will be used to populate the entry's metadata.
		"""
		for instrument, segments in seglists.items():
			if valid is None:
				curr_valid = ()
			else:
				curr_valid = valid[instrument]		
			self.add(LigolwSegmentList(active = segments, instruments = set([instrument]), name = name, version = version, comment = comment, valid = curr_valid))