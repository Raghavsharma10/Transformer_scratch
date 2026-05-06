def get_generated_vcl(self, service_id, version_number):
		"""Display the generated VCL for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/generated_vcl" % (service_id, version_number))
		return FastlyVCL(self, content)