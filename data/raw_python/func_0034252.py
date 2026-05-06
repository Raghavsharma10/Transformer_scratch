def insert_from_segwizard(self, fileobj, instruments, name, version = None, comment = None):
		"""
		Parse the contents of the file object fileobj as a
		segwizard-format segment list, and insert the result as a
		new list of "active" segments into this LigolwSegments
		object.  A new entry will be created in the segment_definer
		table for the segment list, and instruments, name and
		comment are used to populate the entry's metadata.  Note
		that the "valid" segments are left empty, nominally
		indicating that there are no periods of validity.
		"""
		self.add(LigolwSegmentList(active = segmentsUtils.fromsegwizard(fileobj, coltype = LIGOTimeGPS), instruments = instruments, name = name, version = version, comment = comment))