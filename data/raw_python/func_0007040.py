def get_settings(self, service_id, version_number):
		"""Get the settings for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/settings" % (service_id, version_number))
		return FastlySettings(self, content)