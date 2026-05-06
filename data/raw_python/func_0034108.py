def fromfilenames(cls, filenames, coltype=LIGOTimeGPS):
		"""
		Read Cache objects from the files named and concatenate the results into a
		single Cache.
		"""
		cache = cls()
		for filename in filenames:
			cache.extend(cls.fromfile(open(filename), coltype=coltype))
		return cache