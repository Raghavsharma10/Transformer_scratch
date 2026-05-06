def tosegwizard(file, seglist, header = True, coltype = int):
	"""
	Write the segmentlist seglist to the file object file in a
	segwizard compatible format.  If header is True, then the output
	will begin with a comment line containing column names.  The
	segment boundaries will be coerced to type coltype and then passed
	to str() before output.
	"""
	if header:
		print >>file, "# seg\tstart    \tstop     \tduration"
	for n, seg in enumerate(seglist):
		print >>file, "%d\t%s\t%s\t%s" % (n, str(coltype(seg[0])), str(coltype(seg[1])), str(coltype(abs(seg))))