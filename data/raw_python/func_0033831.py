def to_range_strings(seglist):
	"""
	Turn a segment list into a list of range strings as could be parsed
	by from_range_strings().  A typical use for this function is in
	machine-generating configuration files or command lines for other
	programs.

	Example:

	>>> from pycbc_glue.segments import *
	>>> segs = segmentlist([segment(0, 10), segment(35, 35), segment(100, infinity())])
	>>> ",".join(to_range_strings(segs))
	'0:10,35,100:'
	"""
	# preallocate the string list
	ranges = [None] * len(seglist)

	# iterate over segments
	for i, seg in enumerate(seglist):
		if not seg:
			ranges[i] = str(seg[0])
		elif (seg[0] is segments.NegInfinity) and (seg[1] is segments.PosInfinity):
			ranges[i] = ":"
		elif (seg[0] is segments.NegInfinity) and (seg[1] is not segments.PosInfinity):
			ranges[i] = ":%s" % str(seg[1])
		elif (seg[0] is not segments.NegInfinity) and (seg[1] is segments.PosInfinity):
			ranges[i] = "%s:" % str(seg[0])
		elif (seg[0] is not segments.NegInfinity) and (seg[1] is not segments.PosInfinity):
			ranges[i] = "%s:%s" % (str(seg[0]), str(seg[1]))
		else:
			raise ValueError(seg)

	# success
	return ranges