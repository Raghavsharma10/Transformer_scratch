def set_main_vcl(self, service_id, version_number, name):
		"""Set the specified VCL as the main."""
		content = self._fetch("/service/%s/version/%d/vcl/%s/main" % (service_id, version_number, name), method="PUT")
		return FastlyVCL(self, content)