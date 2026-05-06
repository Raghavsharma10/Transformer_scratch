def topfnfile(self, fileobj):
		"""
		write a cache object to filename as a plain text pfn file
		"""
		for entry in self:
			print >>fileobj, entry.path
		fileobj.close()