def get_director(self, service_id, version_number, name):
		"""Get the director for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/director/%s" % (service_id, version_number, name))
		return FastlyDirector(self, content)