def appendColumn(self, name):
		"""
		Append a Column element named "name" to the table.  Returns
		the new child.  Raises ValueError if the table already has
		a column by that name, and KeyError if the validcolumns
		attribute of this table does not contain an entry for a
		column by that name.

		Note that the name string is assumed to be "pre-stripped",
		that is it is the significant portion of the elements Name
		attribute.  The Column element's Name attribute will be
		constructed by pre-pending the stripped Table element's
		name and a colon.

		Example:

		>>> import lsctables
		>>> process_table = lsctables.New(lsctables.ProcessTable, [])
		>>> col = process_table.appendColumn("program")
		>>> col.getAttribute("Name")
		'process:program'
		>>> col.Name
		'program'
		"""
		try:
			self.getColumnByName(name)
			# if we get here the table already has that column
			raise ValueError("duplicate Column '%s'" % name)
		except KeyError:
			pass
		column = Column(AttributesImpl({u"Name": "%s:%s" % (StripTableName(self.tableName), name), u"Type": self.validcolumns[name]}))
		streams = self.getElementsByTagName(ligolw.Stream.tagName)
		if streams:
			self.insertBefore(column, streams[0])
		else:
			self.appendChild(column)
		return column