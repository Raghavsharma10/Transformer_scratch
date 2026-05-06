def create_response_object(self, service_id, version_number, name, status="200", response="OK", content="", request_condition=None, cache_condition=None):
		"""Creates a new Response Object."""
		body = self._formdata({
			"name": name,
			"status": status,
			"response": response,
			"content": content,
			"request_condition": request_condition,
			"cache_condition": cache_condition,
		}, FastlyResponseObject.FIELDS)
		content = self._fetch("/service/%s/version/%d/response_object" % (service_id, version_number), method="POST", body=body)
		return FastlyResponseObject(self, content)