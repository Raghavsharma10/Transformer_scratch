def update_cache_settings(self, service_id, version_number, name_key, **kwargs):
		"""Update a specific cache settings object."""
		body = self._formdata(kwargs, FastlyCacheSettings.FIELDS)
		content = self._fetch("/service/%s/version/%d/cache_settings/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyCacheSettings(self, content)