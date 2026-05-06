def list_response_objects(self, service_id, version_number):
		"""Returns all Response Objects for the specified service and version."""
		content = self._fetch("/service/%s/version/%d/response_object" % (service_id, version_number))
		return map(lambda x: FastlyResponseObject(self, x), content)