def from_range_strings(ranges, boundtype = int):
	"""
	Parse a list of ranges expressed as strings in the form "value" or
	"first:last" into an equivalent pycbc_glue.segments.segmentlist.  In the
	latter case, an empty string for "first" and(or) "last" indicates a
	(semi)infinite range.  A typical use for this function is in
	parsing command line options or entries in configuration files.

	NOTE:  the output is a segmentlist as described by the strings;  if
	the segments in the input file are not coalesced or out of order,
	then thusly shall be the output of this function.  It is
	recommended that this function's output be coalesced before use.

	Example:

	>>> text = "0:10,35,100:"
	>>> from_range_strings(text.split(","))
	[segment(0, 10), segment(35, 35), segment(100, infinity)]
	"""
	# preallocate segmentlist
	segs = segments.segmentlist([None] * len(ranges))

	# iterate over strings
	for i, range in enumerate(ranges):
		parts = range.split(":")
		if len(parts) == 1:
			parts = boundtype(parts[0])
			segs[i] = segments.segment(parts, parts)
			continue
		if len(parts) != 2:
			raise ValueError(range)
		if parts[0] == "":
			parts[0] = segments.NegInfinity
		else:
			parts[0] = boundtype(parts[0])
		if parts[1] == "":
			parts[1] = segments.PosInfinity
		else:
			parts[1] = boundtype(parts[1])
		segs[i] = segments.segment(parts[0], parts[1])

	# success
	return segs