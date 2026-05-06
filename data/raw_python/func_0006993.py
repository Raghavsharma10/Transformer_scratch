def get_condition(self, service_id, version_number, name):
		"""Gets a specified condition."""
		content = self._fetch("/service/%s/version/%d/condition/%s" % (service_id, version_number, name))
		return FastlyCondition(self, content)