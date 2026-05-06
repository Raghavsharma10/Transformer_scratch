def update_condition(self, service_id, version_number, name_key, **kwargs):
		"""Updates the specified condition."""
		body = self._formdata(kwargs, FastlyCondition.FIELDS)
		content = self._fetch("/service/%s/version/%d/condition/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlyCondition(self, content)