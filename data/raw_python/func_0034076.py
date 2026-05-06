def asarray(self):
		"""
		Construct a numpy array from this column.  Note that this
		creates a copy of the data, so modifications made to the
		array will *not* be recorded in the original document.
		"""
		# most codes don't use this feature, this is the only place
		# numpy is used here, and importing numpy can be
		# time-consuming, so we derfer the import until needed.
		import numpy
		try:
			dtype = ligolwtypes.ToNumPyType[self.Type]
		except KeyError as e:
			raise TypeError("cannot determine numpy dtype for Column '%s': %s" % (self.getAttribute("Name"), e))
		return numpy.fromiter(self, dtype = dtype)