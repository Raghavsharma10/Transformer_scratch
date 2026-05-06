def update_response_object(self, service_id, version_number, name_key, **kwargs):
		"""Updates the specified Response Object."""
		body = self._formdata(kwargs, FastlyResponseObject.FIELDS)
		content = self._fetch("/service/%s/version/%d/response_object/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyResponseObject(self, content)