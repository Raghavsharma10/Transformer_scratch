def create_header(self, service_id, version_number, name, destination, source, _type=FastlyHeaderType.RESPONSE, action=FastlyHeaderAction.SET, regex=None, substitution=None, ignore_if_set=None, priority=10, response_condition=None, cache_condition=None, request_condition=None):
		body = self._formdata({
			"name": name,
			"dst": destination,
			"src": source,
			"type": _type,
			"action": action,
			"regex": regex,
			"substitution": substitution,
			"ignore_if_set": ignore_if_set,
			"priority": priority,
			"response_condition": response_condition,
			"request_condition": request_condition,
			"cache_condition": cache_condition,
		}, FastlyHeader.FIELDS)
		"""Creates a new Header object."""
		content = self._fetch("/service/%s/version/%d/header" % (service_id, version_number), method="POST", body=body)
		return FastlyHeader(self, content)