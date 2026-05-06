def get_domain(self, service_id, version_number, name):
		"""Get the domain for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/domain/%s" % (service_id, version_number, name))
		return FastlyDomain(self, content)