def check_domain(self, service_id, version_number, name):
		"""Checks the status of a domain's DNS record. Returns an array of 3 items. The first is the details for the domain. The second is the current CNAME of the domain. The third is a boolean indicating whether or not it has been properly setup to use Fastly."""
		content = self._fetch("/service/%s/version/%d/domain/%s/check" % (service_id, version_number, name))
		return FastlyDomainCheck(self, content)