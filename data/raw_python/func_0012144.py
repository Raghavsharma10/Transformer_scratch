def report(self, fraction=None):
		"""report the total progress for the current stack, optionally given the local fraction completed.
		fraction=None: if given, used as the fraction of the local method so far completed.
		runtimes=None: if given, used as the expected runtimes for the current stack.
		"""
		r = Dict()
		local_key = self.stack_key
		if local_key is None: return {}
		runtimes = self.runtimes()
		for key in self.stack_keys:
			if self.current_times.get(key) is None: 
				self.start(key=key)
			runtime = runtimes.get(key) or self.runtime(key)
			if key == local_key and fraction is not None:
				r[key] = fraction
			elif runtime is not None:
				r[key] = (time() - self.current_times[key]) / runtime
		return r