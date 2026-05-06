def get_service_details(self, service_id):
		"""List detailed information on a specified service."""
		content = self._fetch("/service/%s/details" % service_id)
		return FastlyService(self, content)