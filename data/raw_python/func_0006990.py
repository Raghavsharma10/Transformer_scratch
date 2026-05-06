def delete_cache_settings(self, service_id, version_number, name):
		"""Delete a specific cache settings object."""
		content = self._fetch("/service/%s/version/%d/cache_settings/%s" % (service_id, version_number, name), method="DELETE")
		return self._status(content)