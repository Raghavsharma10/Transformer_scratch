def purge_service_by_key(self, service_id, key):
		"""Purge a particular service by a key."""
		content = self._fetch("/service/%s/purge/%s" % (service_id, key), method="POST")
		return self._status(content)