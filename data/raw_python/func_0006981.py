def list_backends(self, service_id, version_number):
		"""List all backends for a particular service and version."""

		content = self._fetch("/service/%s/version/%d/backend" % (service_id, version_number))
		return map(lambda x: FastlyBackend(self, x), content)