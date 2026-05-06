def Fold(seglist1, seglist2):
	"""
	An iterator that generates the results of taking the intersection
	of seglist1 with each segment in seglist2 in turn.  In each result,
	the segment start and stop values are adjusted to be with respect
	to the start of the corresponding segment in seglist2.  See also
	the segmentlist_range() function.

	This has use in applications that wish to convert ranges of values
	to ranges relative to epoch boundaries.  Below, a list of time
	intervals in hours is converted to a sequence of daily interval
	lists with times relative to midnight.

	Example:

	>>> from pycbc_glue.segments import *
	>>> x = segmentlist([segment(0, 13), segment(14, 20), segment(22, 36)])
	>>> for y in Fold(x, segmentlist_range(0, 48, 24)): print y
	...
	[segment(0, 13), segment(14, 20), segment(22, 24)]
	[segment(0, 12)]
	"""
	for seg in seglist2:
		yield (seglist1 & segments.segmentlist([seg])).shift(-seg[0])