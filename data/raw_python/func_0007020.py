def	purge_url(self, host, path):
		"""Purge an individual URL."""
		content = self._fetch(path, method="PURGE", headers={ "Host": host }) 
		return FastlyPurge(self, content)