def intersects_all(self, other):
		"""
		Returns True if each segmentlist in other intersects the
		corresponding segmentlist in self;  returns False
		if this is not the case, or if other is empty.

		See also:

		.intersects(), .all_intersects(), .all_intersects_all()
		"""
		return all(key in self and self[key].intersects(value) for key, value in other.iteritems()) and bool(other)