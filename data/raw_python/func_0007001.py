def create_director(self, service_id, version_number, 
		name, 
		quorum=75,
		_type=FastlyDirectorType.RANDOM,
		retries=5,
		shield=None):
		"""Create a director for a particular service and version."""
		body = self._formdata({
			"name": name,
			"quorum": quorum,
			"type": _type,
			"retries": retries,
			"shield": shield,

		}, FastlyDirector.FIELDS)
		content = self._fetch("/service/%s/version/%d/director" % (service_id, version_number), method="POST", body=body)
		return FastlyDirector(self, content)