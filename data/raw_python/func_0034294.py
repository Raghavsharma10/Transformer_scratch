def intersects_segment(self, seg):
		"""
		Returns True if any segmentlist in self intersects the
		segment, otherwise returns False.
		"""
		return any(value.intersects_segment(seg) for value in self.itervalues())