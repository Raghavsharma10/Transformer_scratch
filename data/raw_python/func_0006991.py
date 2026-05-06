def list_conditions(self, service_id, version_number):
		"""Gets all conditions for a particular service and version."""
		content = self._fetch("/service/%s/version/%d/condition" % (service_id, version_number))
		return map(lambda x: FastlyCondition(self, x), content)