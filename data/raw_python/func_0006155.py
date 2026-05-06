def needsconnection(self, f):
		''' Decorator used to make sure that the connection has been established. '''

		@wraps(f)
		def wrapper(self, *args, **kwargs):
			if not self.connection:
				self.connect()
			return f(self, *args, **kwargs)
		return wrapper