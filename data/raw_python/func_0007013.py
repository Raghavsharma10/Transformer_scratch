def list_headers(self, service_id, version_number):
		"""Retrieves all Header objects for a particular Version of a Service."""
		content = self._fetch("/service/%s/version/%d/header" % (service_id, version_number))
		return map(lambda x: FastlyHeader(self, x), content)