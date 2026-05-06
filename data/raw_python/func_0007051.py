def delete_user(self, user_id):
		"""Delete a user."""
		content = self._fetch("/user/%s" % user_id, method="DELETE")
		return self._status(content)