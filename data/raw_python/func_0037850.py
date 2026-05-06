def get_current_version(self, increment=None):
		"""
		Return as a string the version of the current state of the
		repository -- a tagged version, if present, or the next version
		based on prior tagged releases.
		"""
		ver = (
			self.get_tagged_version()
			or str(self.get_next_version(increment)) + '.dev0'
		)
		return str(ver)