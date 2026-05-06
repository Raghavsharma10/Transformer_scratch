def S2playground(extent):
	"""
	Return a segmentlist identifying the S2 playground times within the
	interval defined by the segment extent.

	Example:

	>>> from pycbc_glue import segments
	>>> S2playground(segments.segment(874000000, 874010000))
	[segment(874000013, 874000613), segment(874006383, 874006983)]
	"""
	lo = int(extent[0])
	lo -= (lo - 729273613) % 6370
	hi = int(extent[1]) + 1
	return segments.segmentlist(segments.segment(t, t + 600) for t in range(lo, hi, 6370)) & segments.segmentlist([extent])