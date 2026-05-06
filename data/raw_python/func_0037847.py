def reset_less_significant(self, significant_version):
		"""
		Reset to zero all version info less significant than the
		indicated version.

		>>> ver = SummableVersion('3.1.2')
		>>> ver.reset_less_significant(SummableVersion('0.1'))
		>>> str(ver)
		'3.1'
		"""
		def nonzero(x):
			return x != 0
		version_len = 3  # strict versions are always a tuple of 3
		significant_pos = rfind(nonzero, significant_version.version)
		significant_pos = version_len + significant_pos + 1
		self.version = (
			self.version[:significant_pos]
			+ (0,) * (version_len - significant_pos))