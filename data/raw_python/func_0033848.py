def insertBefore(self, newchild, refchild):
		"""
		Insert a new child node before an existing child. It must
		be the case that refchild is a child of this node; if not,
		ValueError is raised. newchild is returned.
		"""
		for i, childNode in enumerate(self.childNodes):
			if childNode is refchild:
				self.childNodes.insert(i, newchild)
				newchild.parentNode = self
				self._verifyChildren(i)
				return newchild
		raise ValueError(refchild)