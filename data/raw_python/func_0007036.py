def delete_service(self, service_id):
		"""Delete a service."""
		content = self._fetch("/service/%s" % service_id, method="DELETE")
		return self._status(content)