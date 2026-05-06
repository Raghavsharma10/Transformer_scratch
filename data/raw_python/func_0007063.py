def update_version(self, service_id, version_number, **kwargs):
		"""Update a particular version for a particular service."""
		body = self._formdata(kwargs, FastlyVersion.FIELDS)
		content = self._fetch("/service/%s/version/%d/" % (service_id, version_number), method="PUT", body=body)
		return FastlyVersion(self, content)