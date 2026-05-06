def check_purge_status(self, purge_id):
		"""Get the status and times of a recently completed purge."""
		content = self._fetch("/purge?id=%s" % purge_id)
		return map(lambda x: FastlyPurgeStatus(self, x), content)