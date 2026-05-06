def get_next_version(self, increment=None):
		"""
		Return the next version based on prior tagged releases.
		"""
		increment = increment or self.increment
		return self.infer_next_version(self.get_latest_version(), increment)