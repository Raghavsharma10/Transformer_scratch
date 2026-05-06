def removeChild(self, child):
		"""
		Remove a child from this element.  The child element is
		returned, and it's parentNode element is reset.
		"""
		super(Table, self).removeChild(child)
		if child.tagName == ligolw.Column.tagName:
			self._update_column_info()
		return child