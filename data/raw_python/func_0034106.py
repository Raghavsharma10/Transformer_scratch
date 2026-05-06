def from_T050017(cls, url, coltype = LIGOTimeGPS):
		"""
		Parse a URL in the style of T050017-00 into a CacheEntry.
		The T050017-00 file name format is, essentially,

		observatory-description-start-duration.extension

		Example:

		>>> c = CacheEntry.from_T050017("file://localhost/data/node144/frames/S5/strain-L2/LLO/L-L1_RDS_C03_L2-8365/L-L1_RDS_C03_L2-836562330-83.gwf")
		>>> c.observatory
		'L'
		>>> c.host
		'localhost'
		>>> os.path.basename(c.path)
		'L-L1_RDS_C03_L2-836562330-83.gwf'
		"""
		match = cls._url_regex.search(url)
		if not match:
			raise ValueError("could not convert %s to CacheEntry" % repr(url))
		observatory = match.group("obs")
		description = match.group("dsc")
		start = match.group("strt")
		duration = match.group("dur")
		if start == "-" and duration == "-":
			# no segment information
			segment = None
		else:
			segment = segments.segment(coltype(start), coltype(start) + coltype(duration))
		return cls(observatory, description, segment, url)