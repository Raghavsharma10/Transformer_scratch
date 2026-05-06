def as_number(self):
		"""
		>>> round(SummableVersion('1.9.3').as_number(), 12)
		1.93
		"""
		def combine(subver, ver):
			return subver / 10 + ver
		return reduce(combine, reversed(self.version))