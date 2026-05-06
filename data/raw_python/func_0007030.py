def create_service(self, customer_id, name, publish_key=None, comment=None):
		"""Create a service."""
		body = self._formdata({
			"customer_id": customer_id,
			"name": name,
			"publish_key": publish_key,
			"comment": comment,
		}, FastlyService.FIELDS)
		content = self._fetch("/service", method="POST", body=body)
		return FastlyService(self, content)