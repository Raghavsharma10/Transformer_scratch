def write(self, fileobj = sys.stdout, indent = u""):
		"""
		Recursively write an element and it's children to a file.
		"""
		fileobj.write(self.start_tag(indent))
		fileobj.write(u"\n")
		for c in self.childNodes:
			if c.tagName not in self.validchildren:
				raise ElementError("invalid child %s for %s" % (c.tagName, self.tagName))
			c.write(fileobj, indent + Indent)
		if self.pcdata is not None:
			fileobj.write(xmlescape(self.pcdata))
			fileobj.write(u"\n")
		fileobj.write(self.end_tag(indent))
		fileobj.write(u"\n")