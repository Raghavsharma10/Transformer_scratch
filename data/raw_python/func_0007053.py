def list_vcls(self, service_id, version_number):
		"""List the uploaded VCLs for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/vcl" % (service_id, version_number))
		return map(lambda x: FastlyVCL(self, x), content)