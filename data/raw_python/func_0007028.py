def get_response_object(self, service_id, version_number, name):
		"""Gets the specified Response Object."""
		content = self._fetch("/service/%s/version/%d/response_object/%s" % (service_id, version_number, name))
		return FastlyResponseObject(self, content)