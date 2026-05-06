def update_settings(self, service_id, version_number, settings={}):
		"""Update the settings for a particular service and version."""
		body = urllib.urlencode(settings)
		content = self._fetch("/service/%s/version/%d/settings" % (service_id, version_number), method="PUT", body=body)
		return FastlySettings(self, content)