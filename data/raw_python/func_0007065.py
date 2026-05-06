def activate_version(self, service_id, version_number):
		"""Activate the current version."""
		content = self._fetch("/service/%s/version/%d/activate" % (service_id, version_number), method="PUT")
		return FastlyVersion(self, content)