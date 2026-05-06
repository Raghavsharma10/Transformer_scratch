def from_gps(cls, gps, Name = None):
		"""
		Instantiate a Time element initialized to the value of the
		given GPS time.  The Name attribute will be set to the
		value of the Name parameter if given.

		Note:  the new Time element holds a reference to the GPS
		time, not a copy of it.  Subsequent modification of the GPS
		time object will be reflected in what gets written to disk.
		"""
		self = cls(AttributesImpl({u"Type": u"GPS"}))
		if Name is not None:
			self.Name = Name
		self.pcdata = gps
		return self