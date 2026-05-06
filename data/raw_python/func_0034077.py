def unlink(self):
		"""
		Break internal references within the document tree rooted
		on this element to promote garbage collection.
		"""
		self._tokenizer = None
		self._rowbuilder = None
		super(TableStream, self).unlink()