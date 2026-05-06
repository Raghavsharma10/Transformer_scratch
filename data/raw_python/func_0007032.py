def get_service(self, service_id):
		"""Get a specific service by id."""
		content = self._fetch("/service/%s" % service_id)
		return FastlyService(self, content)