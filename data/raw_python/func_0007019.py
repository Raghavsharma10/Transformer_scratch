def get_healthcheck(self, service_id, version_number, name):
		"""Get the healthcheck for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/healthcheck/%s" % (service_id, version_number, name))
		return FastlyHealthCheck(self, content)