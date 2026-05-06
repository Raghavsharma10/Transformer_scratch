def list_services(self):
		"""List Services."""
		content = self._fetch("/service")
		return map(lambda x: FastlyService(self, x), content)