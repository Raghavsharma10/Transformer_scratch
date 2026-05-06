def new_param(name, type, value, start = None, scale = None, unit = None, dataunit = None, comment = None):
	"""
	Construct a LIGO Light Weight XML Param document subtree.  FIXME:
	document keyword arguments.
	"""
	elem = Param()
	elem.Name = name
	elem.Type = type
	elem.pcdata = value
	# FIXME:  I have no idea how most of the attributes should be
	# encoded, I don't even know what they're supposed to be.
	if dataunit is not None:
		elem.DataUnit = dataunit
	if scale is not None:
		elem.Scale = scale
	if start is not None:
		elem.Start = start
	if unit is not None:
		elem.Unit = unit
	if comment is not None:
		elem.appendChild(ligolw.Comment())
		elem.childNodes[-1].pcdata = comment
	return elem