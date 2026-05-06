def list_request_settings(self, service_id, version_number):
		"""Returns a list of all Request Settings objects for the given service and version."""
		content = self._fetch("/service/%s/version/%d/request_settings" % (service_id, version_number))
		return map(lambda x: FastlyRequestSetting(self, x), content)