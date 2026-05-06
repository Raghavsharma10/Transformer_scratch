def finish(self):
		"""record the current stack process as finished"""
		self.report(fraction=1.0)
		key = self.stack_key
		if key is not None:
			if self.data.get(key) is None:
				self.data[key] = []
			start_time = self.current_times.get(key) or time()
			self.data[key].append(Dict(runtime=time()-start_time, **self.params))