def _wait_until_exp(self, timeout, error, function, *args):
		"""This replaces the method from Selenium2Library to fix the major logic error in it"""
	
		error = error.replace('<TIMEOUT>', self._format_timeout(timeout))
	
		def wait_func():
			return None if function(*args) else error
			
		self._wait_until_no_error_exp(timeout, wait_func)