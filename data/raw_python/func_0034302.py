def is_coincident(self, other, keys = None):
		"""
		Return True if any segment in any list in self intersects
		any segment in any list in other.  If the optional keys
		argument is not None, then it should be an iterable of keys
		and only segment lists for those keys will be considered in
		the test (instead of raising KeyError, keys not present in
		both segment list dictionaries will be ignored).  If keys
		is None (the default) then all segment lists are
		considered.

		This method is equivalent to the intersects() method, but
		without requiring the keys of the intersecting segment
		lists to match.
		"""
		if keys is not None:
			keys = set(keys)
			self = tuple(self[key] for key in set(self) & keys)
			other = tuple(other[key] for key in set(other) & keys)
		else:
			self = tuple(self.values())
			other = tuple(other.values())
		# make sure inner loop is smallest
		if len(self) < len(other):
			self, other = other, self
		return any(a.intersects(b) for a in self for b in other)