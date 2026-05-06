def get_vcl(self, service_id, version_number, name, include_content=True):
		"""Get the uploaded VCL for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/vcl/%s?include_content=%d" % (service_id, version_number, name, int(include_content)))
		return FastlyVCL(self, content)