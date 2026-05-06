def removeChild(self, child):
		"""
		Remove a child from this element.  The child element is
		returned, and it's parentNode element is reset.  If the
		child will not be used any more, you should call its
		unlink() method to promote garbage collection.
		"""
		for i, childNode in enumerate(self.childNodes):
			if childNode is child:
				del self.childNodes[i]
				child.parentNode = None
				return child
		raise ValueError(child)