def intersects(self, other):
		"""
		Returns True if there exists a segmentlist in self that
		intersects the corresponding segmentlist in other;  returns
		False otherwise.

		See also:

		.intersects_all(), .all_intersects(), .all_intersects_all()
		"""
		return any(key in self and self[key].intersects(value) for key, value in other.iteritems())