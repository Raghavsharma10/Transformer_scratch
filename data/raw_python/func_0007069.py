def list_wordpressess(self, service_id, version_number):
		"""Get all of the wordpresses for a specified service and version."""
		content = self._fetch("/service/%s/version/%d/wordpress" % (service_id, version_number))
		return map(lambda x: FastlyWordpress(self, x), content)