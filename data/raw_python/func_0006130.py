def _needs_download(self, f):
		''' Decorator used to make sure that the downloading happens prior to running the task. '''

		@wraps(f)
		def wrapper(self, *args, **kwargs):
			if not self.isdownloaded():
				self.download()
			return f(self, *args, **kwargs)
		return wrapper