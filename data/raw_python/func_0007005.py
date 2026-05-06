def delete_director_backend(self, service_id, version_number, director_name, backend_name):
		"""Deletes the relationship between a Backend and a Director. The Backend is no longer considered a member of the Director and thus will not have traffic balanced onto it from this Director."""
		content = self._fetch("/service/%s/version/%d/director/%s/backend/%s" % (service_id, version_number, director_name, backend_name), method="DELETE")
		return self._status(content)