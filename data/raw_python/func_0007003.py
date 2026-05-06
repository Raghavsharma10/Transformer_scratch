def update_director(self, service_id, version_number, name_key, **kwargs):
		"""Update the director for a particular service and version."""
		body = self._formdata(kwargs, FastlyDirector.FIELDS)
		content = self._fetch("/service/%s/version/%d/director/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyDirector(self, content)