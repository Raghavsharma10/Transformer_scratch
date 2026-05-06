def update_vcl(self, service_id, version_number, name_key, **kwargs):
		"""Update the uploaded VCL for a particular service and version."""
		body = self._formdata(kwargs, FastlyVCL.FIELDS)
		content = self._fetch("/service/%s/version/%d/vcl/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyVCL(self, content)