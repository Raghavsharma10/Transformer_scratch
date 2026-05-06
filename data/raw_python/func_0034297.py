def all_intersects_all(self, other):
		"""
		Returns True if self and other have the same keys, and each
		segmentlist intersects the corresponding segmentlist in the
		other;  returns False if this is not the case or if either
		dictionary is empty.

		See also:

		.intersects(), .all_intersects(), .intersects_all()
		"""
		return set(self) == set(other) and all(other[key].intersects(value) for key, value in self.iteritems()) and bool(self)