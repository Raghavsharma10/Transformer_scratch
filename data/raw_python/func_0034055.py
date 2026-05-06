def get_time_slide_id(xmldoc, time_slide, create_new = None, superset_ok = False, nonunique_ok = False):
	"""
	Return the time_slide_id corresponding to the offset vector
	described by time_slide, a dictionary of instrument/offset pairs.

	Example:

	>>> get_time_slide_id(xmldoc, {"H1": 0, "L1": 0})
	'time_slide:time_slide_id:10'

	This function is a wrapper around the .get_time_slide_id() method
	of the pycbc_glue.ligolw.lsctables.TimeSlideTable class.  See the
	documentation for that class for the meaning of the create_new,
	superset_ok and nonunique_ok keyword arguments.

	This function requires the document to contain exactly one
	time_slide table.  If the document does not contain exactly one
	time_slide table then ValueError is raised, unless the optional
	create_new argument is not None.  In that case a new table is
	created.  This effect of the create_new argument is in addition to
	the affects described by the TimeSlideTable class.
	"""
	try:
		tisitable = lsctables.TimeSlideTable.get_table(xmldoc)
	except ValueError:
		# table not found
		if create_new is None:
			raise
		tisitable = lsctables.New(lsctables.TimeSlideTable)
		xmldoc.childNodes[0].appendChild(tisitable)
	# make sure the next_id attribute is correct
	tisitable.sync_next_id()
	# get the id
	return tisitable.get_time_slide_id(time_slide, create_new = create_new, superset_ok = superset_ok, nonunique_ok = nonunique_ok)