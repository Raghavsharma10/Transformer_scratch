def get_service_by_name(self, service_name):
		"""Get a specific service by name."""
		content = self._fetch("/service/search?name=%s" % service_name)
		return FastlyService(self, content)