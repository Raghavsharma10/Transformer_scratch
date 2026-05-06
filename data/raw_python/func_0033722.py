def deltas(self):
		"""
		Dictionary of relative offsets.  The keys in the result are
		pairs of keys from the offset vector, (a, b), and the
		values are the relative offsets, (offset[b] - offset[a]).
		Raises ValueError if the offsetvector is empty (WARNING:
		this behaviour might change in the future).

		Example:

		>>> x = offsetvector({"H1": 0, "L1": 10, "V1": 20})
		>>> x.deltas
		{('H1', 'L1'): 10, ('H1', 'V1'): 20, ('H1', 'H1'): 0}
		>>> y = offsetvector({'H1': 100, 'L1': 110, 'V1': 120})
		>>> y.deltas == x.deltas
		True

		Note that the result always includes a "dummy" entry,
		giving the relative offset of self.refkey with respect to
		itself, which is always 0.

		See also .fromdeltas().

		BUGS:  I think the keys in each tuple should be reversed.
		I can't remember why I put them in the way they are.
		Expect them to change in the future.
		"""
		# FIXME:  instead of raising ValueError when the
		# offsetvector is empty this should return an empty
		# dictionary.  the inverse, .fromdeltas() accepts
		# empty dictionaries
		# NOTE:  the arithmetic used to construct the offsets
		# *must* match the arithmetic used by
		# time_slide_component_vectors() so that the results of the
		# two functions can be compared to each other without worry
		# of floating-point round off confusing things.
		refkey = self.refkey
		refoffset = self[refkey]
		return dict(((refkey, key), self[key] - refoffset) for key in self)