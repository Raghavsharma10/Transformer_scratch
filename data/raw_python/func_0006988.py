def get_cache_settings(self, service_id, version_number, name):
		"""Get a specific cache settings object."""
		content = self._fetch("/service/%s/version/%d/cache_settings/%s" % (service_id, version_number, name))
		return FastlyCacheSettings(self, content)