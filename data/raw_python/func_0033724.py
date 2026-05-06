def normalize(self, **kwargs):
		"""
		Adjust the offsetvector so that a particular instrument has
		the desired offset.  All other instruments have their
		offsets adjusted so that the relative offsets are
		preserved.  The instrument to noramlize, and the offset one
		wishes it to have, are provided as a key-word argument.
		The return value is the time slide dictionary, which is
		modified in place.

		If more than one key-word argument is provided the keys are
		sorted and considered in order until a key is found that is
		in the offset vector.  The offset vector is normalized to
		that value.  This function is a no-op if no key-word
		argument is found that applies.

		Example:

		>>> a = offsetvector({"H1": -10, "H2": -10, "L1": -10})
		>>> a.normalize(L1 = 0)
		offsetvector({'H2': 0, 'H1': 0, 'L1': 0})
		>>> a = offsetvector({"H1": -10, "H2": -10})
		>>> a.normalize(L1 = 0, H2 = 5)
		offsetvector({'H2': 5, 'H1': 5})
		"""
		# FIXME:  should it be performed in place?  if it should
		# be, the should there be no return value?
		for key, offset in sorted(kwargs.items()):
			if key in self:
				delta = offset - self[key]
				for key in self.keys():
					self[key] += delta
				break
		return self