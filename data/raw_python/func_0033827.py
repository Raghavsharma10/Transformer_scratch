def fromsegwizard(file, coltype = int, strict = True):
	"""
	Read a segmentlist from the file object file containing a segwizard
	compatible segment list.  Parsing stops on the first line that
	cannot be parsed (which is consumed).  The segmentlist will be
	created with segment whose boundaries are of type coltype, which
	should raise ValueError if it cannot convert its string argument.
	Two-column, three-column, and four-column segwizard files are
	recognized, but the entire file must be in the same format, which
	is decided by the first parsed line.  If strict is True and the
	file is in three- or four-column format, then each segment's
	duration is checked against that column in the input file.

	NOTE:  the output is a segmentlist as described by the file;  if
	the segments in the input file are not coalesced or out of order,
	then thusly shall be the output of this function.  It is
	recommended that this function's output be coalesced before use.
	"""
	commentpat = re.compile(r"\s*([#;].*)?\Z", re.DOTALL)
	twocolsegpat = re.compile(r"\A\s*([\d.+-eE]+)\s+([\d.+-eE]+)\s*\Z")
	threecolsegpat = re.compile(r"\A\s*([\d.+-eE]+)\s+([\d.+-eE]+)\s+([\d.+-eE]+)\s*\Z")
	fourcolsegpat = re.compile(r"\A\s*([\d]+)\s+([\d.+-eE]+)\s+([\d.+-eE]+)\s+([\d.+-eE]+)\s*\Z")
	format = None
	l = segments.segmentlist()
	for line in file:
		line = commentpat.split(line)[0]
		if not line:
			continue
		try:
			[tokens] = fourcolsegpat.findall(line)
			num = int(tokens[0])
			seg = segments.segment(map(coltype, tokens[1:3]))
			duration = coltype(tokens[3])
			this_line_format = 4
		except ValueError:
			try:
				[tokens] = threecolsegpat.findall(line)
				seg = segments.segment(map(coltype, tokens[0:2]))
				duration = coltype(tokens[2])
				this_line_format = 3
			except ValueError:
				try:
					[tokens] = twocolsegpat.findall(line)
					seg = segments.segment(map(coltype, tokens[0:2]))
					duration = abs(seg)
					this_line_format = 2
				except ValueError:
					break
		if strict:
			if abs(seg) != duration:
				raise ValueError("segment '%s' has incorrect duration" % line)
			if format is None:
				format = this_line_format
			elif format != this_line_format:
				raise ValueError("segment '%s' format mismatch" % line)
		l.append(seg)
	return l