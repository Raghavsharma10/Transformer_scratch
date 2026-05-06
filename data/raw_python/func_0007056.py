def get_vcl_html(self, service_id, version_number, name):
		"""Get the uploaded VCL for a particular service and version with HTML syntax highlighting."""
		content = self._fetch("/service/%s/version/%d/vcl/%s/content" % (service_id, version_number, name))
		return content.get("content", None)