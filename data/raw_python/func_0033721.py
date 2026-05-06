def component_offsetvectors(offsetvectors, n):
	"""
	Given an iterable of offset vectors, return the shortest list of
	the unique n-instrument offset vectors from which all the vectors
	in the input iterable can be constructed.  This can be used to
	determine the minimal set of n-instrument coincs required to
	construct all of the coincs for all of the requested instrument and
	offset combinations in a set of offset vectors.

	It is assumed that the coincs for the vector {"H1": 0, "H2": 10,
	"L1": 20} can be constructed from the coincs for the vectors {"H1":
	0, "H2": 10} and {"H2": 0, "L1": 10}, that is only the relative
	offsets are significant in determining if two events are
	coincident, not the absolute offsets.  This assumption is not true
	for the standard inspiral pipeline, where the absolute offsets are
	significant due to the periodic wrapping of triggers around rings.
	"""
	#
	# collect unique instrument set / deltas combinations
	#

	delta_sets = {}
	for vect in offsetvectors:
		for instruments in iterutils.choices(sorted(vect), n):
			# NOTE:  the arithmetic used to construct the
			# offsets *must* match the arithmetic used by
			# offsetvector.deltas so that the results of the
			# two can be compared to each other without worry
			# of floating-point round off confusing things.
			delta_sets.setdefault(instruments, set()).add(tuple(vect[instrument] - vect[instruments[0]] for instrument in instruments))

	#
	# translate into a list of normalized n-instrument offset vectors
	#

	return [offsetvector(zip(instruments, deltas)) for instruments, delta_set in delta_sets.items() for deltas in delta_set]