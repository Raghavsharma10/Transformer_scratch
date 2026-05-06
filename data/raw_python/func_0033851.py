def replaceChild(self, newchild, oldchild):
		"""
		Replace an existing node with a new node. It must be the
		case that oldchild is a child of this node; if not,
		ValueError is raised. newchild is returned.
		"""
		# .index() would use compare-by-value, we want
		# compare-by-id because we want to find the exact object,
		# not something equivalent to it.
		for i, childNode in enumerate(self.childNodes):
			if childNode is oldchild:
				self.childNodes[i].parentNode = None
				self.childNodes[i] = newchild
				newchild.parentNode = self
				self._verifyChildren(i)
				return newchild
		raise ValueError(oldchild)