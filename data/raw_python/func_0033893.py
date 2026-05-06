def get_shape(self):
		"""
		Return a tuple of this array's dimensions.  This is done by
		querying the Dim children.  Note that once it has been
		created, it is also possible to examine an Array object's
		.array attribute directly, and doing that is much faster.
		"""
		return tuple(int(c.pcdata) for c in self.getElementsByTagName(ligolw.Dim.tagName))[::-1]