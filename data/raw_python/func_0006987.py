def create_cache_settings(self, 
		service_id, 
		version_number, 
		name,
		action,
		ttl=None,
		stale_ttl=None,
		cache_condition=None):
		"""Create a new cache settings object."""
		body = self._formdata({
			"name": name,
			"action": action,
			"ttl": ttl,
			"stale_ttl": stale_ttl,
			"cache_condition": cache_condition,
		}, FastlyCacheSettings.FIELDS)
		content = self._fetch("/service/%s/version/%d/cache_settings" % (service_id, version_number), method="POST", body=body)
		return FastlyCacheSettings(self, content)