def now(cls, Name = None):
		"""
		Instantiate a Time element initialized to the current UTC
		time in the default format (ISO-8601).  The Name attribute
		will be set to the value of the Name parameter if given.
		"""
		self = cls()
		if Name is not None:
			self.Name = Name
		self.pcdata = datetime.datetime.utcnow()
		return self