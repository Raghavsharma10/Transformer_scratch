def list_domains(self, service_id, version_number):
		"""List the domains for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/domain" % (service_id, version_number))
		return map(lambda x: FastlyDomain(self, x), content)