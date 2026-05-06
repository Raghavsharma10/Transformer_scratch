def purge_service(self, service_id):
		"""Purge everything from a service."""
		content = self._fetch("/service/%s/purge_all" % service_id, method="POST")
		return self._status(content)