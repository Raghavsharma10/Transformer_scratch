def get_event_log(self, object_id):
		"""Get the specified event log."""
		content = self._fetch("/event_log/%s" % object_id, method="GET")
		return FastlyEventLog(self, content)