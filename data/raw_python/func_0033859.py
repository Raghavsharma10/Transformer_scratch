def write(self, fileobj = sys.stdout, xsl_file = None):
		"""
		Write the document.
		"""
		fileobj.write(Header)
		fileobj.write(u"\n")
		if xsl_file is not None:
			fileobj.write(u'<?xml-stylesheet type="text/xsl" href="%s" ?>\n' % xsl_file)
		for c in self.childNodes:
			if c.tagName not in self.validchildren:
				raise ElementError("invalid child %s for %s" % (c.tagName, self.tagName))
			c.write(fileobj)