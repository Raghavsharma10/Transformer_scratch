def update_wordpress(self, service_id, version_number, name_key, **kwargs):
		"""Update a specified wordpress."""
		body = self._formdata(kwargs, FastlyWordpress.FIELDS)
		content = self._fetch("/service/%s/version/%d/wordpress/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyWordpress(self, content)