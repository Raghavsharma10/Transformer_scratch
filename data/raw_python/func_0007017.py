def list_healthchecks(self, service_id, version_number):
		"""List all of the healthchecks for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/healthcheck" % (service_id, version_number))
		return map(lambda x: FastlyHealthCheck(self, x), content)