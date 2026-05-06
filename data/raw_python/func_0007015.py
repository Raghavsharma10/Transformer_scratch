def get_header(self, service_id, version_number, name):
		"""Retrieves a Header object by name."""
		content = self._fetch("/service/%s/version/%d/header/%s" % (service_id, version_number, name))
		return FastlyHeader(self, content)