def write(self, fileobj = sys.stdout, indent = u""):
		"""
		Recursively write an element and it's children to a file.
		"""
		fileobj.write(self.start_tag(indent))
		fileobj.write(u"\n")