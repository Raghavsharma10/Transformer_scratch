def fromdeltas(cls, deltas):
		"""
		Construct an offsetvector from a dictionary of offset
		deltas as returned by the .deltas attribute.

		Example:

		>>> x = offsetvector({"H1": 0, "L1": 10, "V1": 20})
		>>> y = offsetvector.fromdeltas(x.deltas)
		>>> y
		offsetvector({'V1': 20, 'H1': 0, 'L1': 10})
		>>> y == x
		True

		See also .deltas, .fromkeys()
		"""
		return cls((key, value) for (refkey, key), value in deltas.items())