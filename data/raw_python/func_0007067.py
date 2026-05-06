def validate_version(self, service_id, version_number):
		"""Validate the version for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/validate" % (service_id, version_number))
		return self._status(content)