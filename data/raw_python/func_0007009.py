def update_domain(self, service_id, version_number, name_key, **kwargs):
		"""Update the domain for a particular service and version."""
		body = self._formdata(kwargs, FastlyDomain.FIELDS)
		content = self._fetch("/service/%s/version/%d/domain/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyDomain(self, content)