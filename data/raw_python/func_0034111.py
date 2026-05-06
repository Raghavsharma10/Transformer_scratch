def tofile(self, fileobj):
		"""
		write a cache object to the fileobj as a lal cache file
		"""
		for entry in self:
			print >>fileobj, str(entry)
		fileobj.close()