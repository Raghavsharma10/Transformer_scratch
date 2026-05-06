def _update_column_info(self):
		"""
		Used for validation during parsing, and additional
		book-keeping.  For internal use only.
		"""
		del self.columnnames[:]
		del self.columntypes[:]
		del self.columnpytypes[:]
		for child in self.getElementsByTagName(ligolw.Column.tagName):
			if self.validcolumns is not None:
				try:
					if self.validcolumns[child.Name] != child.Type:
						raise ligolw.ElementError("invalid type '%s' for Column '%s' in Table '%s', expected type '%s'" % (child.Type, child.getAttribute("Name"), self.getAttribute("Name"), self.validcolumns[child.Name]))
				except KeyError:
					raise ligolw.ElementError("invalid Column '%s' for Table '%s'" % (child.getAttribute("Name"), self.getAttribute("Name")))
			if child.Name in self.columnnames:
				raise ligolw.ElementError("duplicate Column '%s' in Table '%s'" % (child.getAttribute("Name"), self.getAttribute("Name")))
			self.columnnames.append(child.Name)
			self.columntypes.append(child.Type)
			try:
				self.columnpytypes.append(ligolwtypes.ToPyType[child.Type])
			except KeyError:
				raise ligolw.ElementError("unrecognized Type '%s' for Column '%s' in Table '%s'" % (child.Type, child.getAttribute("Name"), self.getAttribute("Name")))