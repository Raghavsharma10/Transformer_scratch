def create_condition(self, 
		service_id, 
		version_number,
		name,
		_type,
		statement,
		priority="10", 
		comment=None):
		"""Creates a new condition."""
		body = self._formdata({
			"name": name,
			"type": _type,
			"statement": statement,
			"priority": priority,
			"comment": comment,
		}, FastlyCondition.FIELDS)
		content = self._fetch("/service/%s/version/%d/condition" % (service_id, version_number), method="POST", body=body)
		return FastlyCondition(self, content)