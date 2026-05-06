def time_slides_vacuum(time_slides, verbose = False):
	"""
	Given a dictionary mapping time slide IDs to instrument-->offset
	mappings, for example as returned by the as_dict() method of the
	TimeSlideTable class in pycbc_glue.ligolw.lsctables or by the
	load_time_slides() function in this module, construct and return a
	mapping indicating time slide equivalences.  This can be used to
	delete redundant time slides from a time slide table, and then also
	used via the applyKeyMapping() method of pycbc_glue.ligolw.table.Table
	instances to update cross references (for example in the
	coinc_event table).

	Example:

	>>> slides = {"time_slide_id:0": {"H1": 0, "H2": 0},
	"time_slide_id:1": {"H1": 10, "H2": 10}, "time_slide_id:2": {"H1":
	0, "H2": 10}}
	>>> time_slides_vacuum(slides)
	{'time_slide_id:1': 'time_slide_id:0'}

	indicating that time_slide_id:1 describes a time slide that is
	equivalent to time_slide_id:0.  The calling code could use this
	information to delete time_slide_id:1 from the time_slide table,
	and replace references to that ID in other tables with references
	to time_slide_id:0.
	"""
	# convert offsets to deltas
	time_slides = dict((time_slide_id, offsetvect.deltas) for time_slide_id, offsetvect in time_slides.items())
	if verbose:
		progressbar = ProgressBar(max = len(time_slides))
	else:
		progressbar = None
	# old --> new mapping
	mapping = {}
	# while there are time slide offset dictionaries remaining
	while time_slides:
		# pick an ID/offset dictionary pair at random
		id1, deltas1 = time_slides.popitem()
		# for every other ID/offset dictionary pair in the time
		# slides
		ids_to_delete = []
		for id2, deltas2 in time_slides.items():
			# if the relative offset dictionaries are
			# equivalent record in the old --> new mapping
			if deltas2 == deltas1:
				mapping[id2] = id1
				ids_to_delete.append(id2)
		for id2 in ids_to_delete:
			time_slides.pop(id2)
		if progressbar is not None:
			progressbar.update(progressbar.max - len(time_slides))
	# done
	del progressbar
	return mapping