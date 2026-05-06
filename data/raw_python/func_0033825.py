def fromfilenames(filenames, coltype = int):
	"""
	Return a segmentlist describing the intervals spanned by the files
	whose names are given in the list filenames.  The segmentlist is
	constructed by parsing the file names, and the boundaries of each
	segment are coerced to type coltype.

	The file names are parsed using a generalization of the format
	described in Technical Note LIGO-T010150-00-E, which allows the
	start time and duration appearing in the file name to be
	non-integers.

	NOTE:  the output is a segmentlist as described by the file names;
	if the file names are not in time order, or describe overlaping
	segments, then thusly shall be the output of this function.  It is
	recommended that this function's output be coalesced before use.
	"""
	pattern = re.compile(r"-([\d.]+)-([\d.]+)\.[\w_+#]+\Z")
	l = segments.segmentlist()
	for name in filenames:
		[(s, d)] = pattern.findall(name.strip().rstrip(".gz"))
		s = coltype(s)
		d = coltype(d)
		l.append(segments.segment(s, s + d))
	return l