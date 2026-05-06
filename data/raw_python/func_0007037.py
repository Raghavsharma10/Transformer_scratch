def list_domains_by_service(self, service_id):
		"""List the domains within a service."""
		content = self._fetch("/service/%s/domain" % service_id, method="GET")
		return map(lambda x: FastlyDomain(self, x), content)