def get_wordpress(self, service_id, version_number, name):
		"""Get information on a specific wordpress."""
		content = self._fetch("/service/%s/version/%d/wordpress/%s" % (service_id, version_number, name))
		return FastlyWordpress(self, content)