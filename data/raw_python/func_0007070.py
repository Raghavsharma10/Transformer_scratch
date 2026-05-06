def create_wordpress(self,
		service_id,
		version_number,
		name,
		path,
		comment=None):
		"""Create a wordpress for the specified service and version."""
		body = self._formdata({
			"name": name,
			"path": path,
			"comment": comment,
		}, FastlyWordpress.FIELDS)
		content = self._fetch("/service/%s/version/%d/wordpress" % (service_id, version_number), method="POST", body=body)
		return FastlyWordpress(self, content)