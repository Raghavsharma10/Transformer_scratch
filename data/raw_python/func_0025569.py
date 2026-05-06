def prepare(self, context):
		"""Executed prior to processing a request."""
		if __debug__:
			log.debug("Assigning thread local request context.")
		
		self.local.context = context