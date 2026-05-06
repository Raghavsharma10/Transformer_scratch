def _needs_elements(self, f):
		''' Decorator used to make sure that there are elements prior to running the task. '''

		@wraps(f)
		def wrapper(self, *args, **kwargs):
			if self.elements == None:
				self.getelements()
			return f(self, *args, **kwargs)
		return wrapper