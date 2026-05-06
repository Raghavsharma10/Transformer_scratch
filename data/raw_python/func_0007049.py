def create_user(self, customer_id, name, login, password, role=FastlyRoles.USER, require_new_password=True):
		"""Create a user."""
		body = self._formdata({
			"customer_id": customer_id,
			"name": name,
			"login": login,
			"password": password,
			"role": role,
			"require_new_password": require_new_password,
		}, FastlyUser.FIELDS)
		content = self._fetch("/user", method="POST", body=body)
		return FastlyUser(self, content)