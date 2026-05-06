def get_version(self, service_id, version_number):
		"""Get the version for a particular service."""
		content = self._fetch("/service/%s/version/%d" % (service_id, version_number))
		return FastlyVersion(self, content)