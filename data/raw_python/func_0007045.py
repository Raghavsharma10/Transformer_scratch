def get_syslog(self, service_id, version_number, name):
		"""Get the Syslog for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/syslog/%s" % (service_id, version_number, name))
		return FastlySyslog(self, content)