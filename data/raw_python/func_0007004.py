def get_director_backend(self, service_id, version_number, director_name, backend_name):
		"""Returns the relationship between a Backend and a Director. If the Backend has been associated with the Director, it returns a simple record indicating this. Otherwise, returns a 404."""
		content = self._fetch("/service/%s/version/%d/director/%s/backend/%s" % (service_id, version_number, director_name, backend_name), method="GET")
		return FastlyDirectorBackend(self, content)