def list_customer_users(self, customer_id):
		"""List all users from a specified customer id."""
		content = self._fetch("/customer/users/%s" % customer_id)
		return map(lambda x: FastlyUser(self, x), content)