def fromfile(cls, fileobj, coltype=LIGOTimeGPS):
		"""
		Return a Cache object whose entries are read from an open file.
		"""
		c = [cls.entry_class(line, coltype=coltype) for line in fileobj]
		return cls(c)