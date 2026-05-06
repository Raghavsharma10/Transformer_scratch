def get_error(self, block = True, timeout = None):
		"""
		Gets the next error message.
		
		Each error message is a 2-tuple of (status, identifier)."""
		return self._error_queue.get(block = block, timeout = timeout)