def list_directors(self, service_id, version_number):
		"""List the directors for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/director" % (service_id, version_number))
		return map(lambda x: FastlyDirector(self, x), content)