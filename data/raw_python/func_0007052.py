def request_password_reset(self, user_id):
		"""Requests a password reset for the specified user."""
		content = self._fetch("/user/%s/password/request_reset" % (user_id), method="POST")
		return FastlyUser(self, content)