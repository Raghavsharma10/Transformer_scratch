def clone_version(self, service_id, version_number):
		"""Clone the current configuration into a new version."""
		content = self._fetch("/service/%s/version/%d/clone" % (service_id, version_number), method="PUT")
		return FastlyVersion(self, content)