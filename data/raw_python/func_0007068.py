def lock_version(self, service_id, version_number):
		"""Locks the specified version."""
		content = self._fetch("/service/%s/version/%d/lock" % (service_id, version_number))
		return self._status(content)