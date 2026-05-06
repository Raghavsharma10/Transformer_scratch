def unlink(self):
		"""
		Break internal references within the document tree rooted
		on this element to promote garbage collection.
		"""
		self.parentNode = None
		for child in self.childNodes:
			child.unlink()
		del self.childNodes[:]