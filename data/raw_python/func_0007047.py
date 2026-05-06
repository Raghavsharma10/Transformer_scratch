def change_password(self, old_password, new_password):
		"""Update the user's password to a new one."""
		body = self._formdata({
			"old_password": old_password,
			"password": new_password,
		}, ["old_password", "password"])
		content = self._fetch("/current_user/password", method="POST", body=body)
		return FastlyUser(self, content)