def time_slide_list_merge(slides1, slides2):
	"""
	Merges two lists of offset dictionaries into a single list with
	no duplicate (equivalent) time slides.
	"""
	deltas1 = set(frozenset(offsetvect1.deltas.items()) for offsetvect1 in slides1)
	return slides1 + [offsetvect2 for offsetvect2 in slides2 if frozenset(offsetvect2.deltas.items()) not in deltas1]