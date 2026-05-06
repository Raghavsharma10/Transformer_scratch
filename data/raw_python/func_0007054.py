def upload_vcl(self, service_id, version_number, name, content, main=None, comment=None):
		"""Upload a VCL for a particular service and version."""
		body = self._formdata({
			"name": name,
			"content": content,
			"comment": comment,
			"main": main,
		}, FastlyVCL.FIELDS)
		content = self._fetch("/service/%s/version/%d/vcl" % (service_id, version_number), method="POST", body=body)
		return FastlyVCL(self, content)