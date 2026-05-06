def unique(self):
		"""
		Return a Cache which has every element of self, but without
		duplication.  Preserve order.  Does not hash, so a bit slow.
		"""
		new = self.__class__([])
		for elem in self:
			if elem not in new:
				new.append(elem)
		return new