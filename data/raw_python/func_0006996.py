def get_customer(self, customer_id):
		"""Get a specific customer."""
		content = self._fetch("/customer/%s" % customer_id)
		return FastlyCustomer(self, content)