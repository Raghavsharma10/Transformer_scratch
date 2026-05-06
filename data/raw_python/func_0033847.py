def appendChild(self, child):
		"""
		Add a child to this element.  The child's parentNode
		attribute is updated, too.
		"""
		self.childNodes.append(child)
		child.parentNode = self
		self._verifyChildren(len(self.childNodes) - 1)
		return child