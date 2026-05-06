def list_syslogs(self, service_id, version_number):
		"""List all of the Syslogs for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/syslog" % (service_id, version_number))
		return map(lambda x: FastlySyslog(self, x), content)