def check_backends(self, service_id, version_number):
		"""Performs a health check against each backend in version. If the backend has a specific type of healthcheck, that one is performed, otherwise a HEAD request to / is performed. The first item is the details on the Backend itself. The second item is details of the specific HTTP request performed as a health check. The third item is the response details."""
		content = self._fetch("/service/%s/version/%d/backend/check_all" % (service_id, version_number))
		# TODO: Use a strong-typed class for output?
		return content