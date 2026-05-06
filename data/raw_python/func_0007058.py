def get_generated_vcl_html(self, service_id, version_number):
		"""Display the content of generated VCL with HTML syntax highlighting."""
		content = self._fetch("/service/%s/version/%d/generated_vcl/content" % (service_id, version_number))
		return content.get("content", None)