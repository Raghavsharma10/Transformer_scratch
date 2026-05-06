def _verifyChildren(self, i):
		"""
		Used for validation during parsing, and additional
		book-keeping.  For internal use only.
		"""
		super(Table, self)._verifyChildren(i)
		child = self.childNodes[i]
		if child.tagName == ligolw.Column.tagName:
			self._update_column_info()
		elif child.tagName == ligolw.Stream.tagName:
			# require agreement of non-stripped strings
			if child.getAttribute("Name") != self.getAttribute("Name"):
				raise ligolw.ElementError("Stream name '%s' does not match Table name '%s'" % (child.getAttribute("Name"), self.getAttribute("Name")))