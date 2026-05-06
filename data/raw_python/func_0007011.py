def check_domains(self, service_id, version_number):
		"""Checks the status of all domain DNS records for a Service Version. Returns an array items in the same format as the single domain /check."""
		content = self._fetch("/service/%s/version/%d/domain/check_all" % (service_id, version_number))
		return map(lambda x: FastlyDomainCheck(self, x), content)