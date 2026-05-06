def get_user(self, user_id):
		"""Get a specific user."""
		content = self._fetch("/user/%s" % user_id)
		return FastlyUser(self, content)