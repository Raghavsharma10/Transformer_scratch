def update_service(self, service_id, **kwargs):
		"""Update a service."""
		body = self._formdata(kwargs, FastlyService.FIELDS)
		content = self._fetch("/service/%s" % service_id, method="PUT", body=body)
		return FastlyService(self, content)