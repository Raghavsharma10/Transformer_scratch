def update_header(self, service_id, version_number, name_key, **kwargs):
		"""Modifies an existing Header object by name."""
		body = self._formdata(kwargs, FastlyHeader.FIELDS)
		content = self._fetch("/service/%s/version/%d/header/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyHeader(self, content)