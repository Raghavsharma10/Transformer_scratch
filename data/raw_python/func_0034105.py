def segmentlistdict(self):
		"""
		A segmentlistdict object describing the instruments and
		time spanned by this CacheEntry.  A new object is
		constructed each time this attribute is accessed (segments
		are immutable so there is no reason to try to share a
		reference to the CacheEntry's internal segment;
		modifications of one would not be reflected in the other
		anyway).

		Example:

		>>> c = CacheEntry(u"H1 S5 815901601 576.5 file://localhost/home/kipp/tmp/1/H1-815901601-576.xml")
		>>> c.segmentlistdict
		{u'H1': [segment(LIGOTimeGPS(815901601, 0), LIGOTimeGPS(815902177, 500000000))]}

		The \"observatory\" column of the cache entry, which is
		frequently used to store instrument names, is parsed into
		instrument names for the dictionary keys using the same
		rules as pycbc_glue.ligolw.lsctables.instrument_set_from_ifos().

		Example:

		>>> c = CacheEntry(u"H1H2, S5 815901601 576.5 file://localhost/home/kipp/tmp/1/H1H2-815901601-576.xml")
		>>> c.segmentlistdict
		{u'H1H2': [segment(LIGOTimeGPS(815901601, 0), LIGOTimeGPS(815902177, 500000000))]}
		"""
		# the import has to be done here to break the cyclic
		# dependancy
		from pycbc_glue.ligolw.lsctables import instrument_set_from_ifos
		instruments = instrument_set_from_ifos(self.observatory) or (None,)
		return segments.segmentlistdict((instrument, segments.segmentlist(self.segment is not None and [self.segment] or [])) for instrument in instruments)