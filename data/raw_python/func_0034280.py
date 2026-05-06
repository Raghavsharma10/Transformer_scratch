def shift(self, x):
		"""
		Return a new segment whose bounds are given by adding x to
		the segment's upper and lower bounds.
		"""
		return tuple.__new__(self.__class__, (self[0] + x, self[1] + x))