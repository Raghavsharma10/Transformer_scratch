def update_backend(self, service_id, version_number, name_key, **kwargs):
		"""Update the backend for a particular service and version."""
		body = self._formdata(kwargs, FastlyBackend.FIELDS)
		content = self._fetch("/service/%s/version/%d/backend/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyBackend(self, content)