def create_version(self, service_id, inherit_service_id=None, comment=None):
		"""Create a version for a particular service."""
		body = self._formdata({
			"service_id": service_id,
			"inherit_service_id": inherit_service_id,
			"comment": comment,
		}, FastlyVersion.FIELDS)
		content = self._fetch("/service/%s/version" % service_id, method="POST", body=body)
		return FastlyVersion(self, content)