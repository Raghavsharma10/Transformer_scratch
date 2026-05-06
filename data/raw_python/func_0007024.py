def get_request_setting(self, service_id, version_number, name):
		"""Gets the specified Request Settings object."""
		content = self._fetch("/service/%s/version/%d/request_settings/%s" % (service_id, version_number, name))
		return FastlyRequestSetting(self, content)