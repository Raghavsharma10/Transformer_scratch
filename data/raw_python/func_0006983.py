def get_backend(self, service_id, version_number, name):
		"""Get the backend for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/backend/%s" % (service_id, version_number, name))
		return FastlyBackend(self, content)