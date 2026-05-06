def getFields(cls) :
		"""returns a set of the available fields. In order to be able ti securely loop of the fields, "raba_id" and "json" are not included in the set"""
		s = set(cls.columns.keys())
		s.remove('json')
		s.remove('raba_id')
		return s