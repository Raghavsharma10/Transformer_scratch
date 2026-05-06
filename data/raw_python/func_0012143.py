def start(self, key=None, **params):
		"""initialize process timing for the current stack"""
		self.params.update(**params)
		key = key or self.stack_key
		if key is not None:
			self.current_times[key] = time()