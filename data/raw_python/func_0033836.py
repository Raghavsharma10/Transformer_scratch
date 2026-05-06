def segmentlist_range(start, stop, period):
	"""
	Analogous to Python's range() builtin, this generator yields a
	sequence of continuous adjacent segments each of length "period"
	with the first starting at "start" and the last ending not after
	"stop".  Note that the segments generated do not form a coalesced
	list (they are not disjoint).  start, stop, and period can be any
	objects which support basic arithmetic operations.

	Example:

	>>> from pycbc_glue.segments import *
	>>> segmentlist(segmentlist_range(0, 15, 5))
	[segment(0, 5), segment(5, 10), segment(10, 15)]
	>>> segmentlist(segmentlist_range('', 'xxx', 'x'))
	[segment('', 'x'), segment('x', 'xx'), segment('xx', 'xxx')]
	"""
	n = 1
	b = start
	while True:
		a, b = b, start + n * period
		if b > stop:
			break
		yield segments.segment(a, b)
		n += 1