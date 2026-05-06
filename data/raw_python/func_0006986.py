def list_cache_settings(self, service_id, version_number):
		"""Get a list of all cache settings for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/cache_settings" % (service_id, version_number))
		return map(lambda x: FastlyCacheSettings(self, x), content)