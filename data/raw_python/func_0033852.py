def getElements(self, filter):
		"""
		Return a list of elements below and including this element
		for which filter(element) returns True.
		"""
		l = reduce(lambda l, e: l + e.getElements(filter), self.childNodes, [])
		if filter(self):
			l.append(self)
		return l