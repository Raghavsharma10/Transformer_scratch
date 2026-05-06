def update_request_setting(self, service_id, version_number, name_key, **kwargs):
		"""Updates the specified Request Settings object."""
		body = self._formdata(kwargs, FastlyHealthCheck.FIELDS)
		content = self._fetch("/service/%s/version/%d/request_settings/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyRequestSetting(self, content)