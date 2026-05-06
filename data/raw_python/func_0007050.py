def update_user(self, user_id, **kwargs):
		"""Update a user."""
		body = self._formdata(kwargs, FastlyUser.FIELDS)
		content = self._fetch("/user/%s" % user_id, method="PUT", body=body)
		return FastlyUser(self, content)