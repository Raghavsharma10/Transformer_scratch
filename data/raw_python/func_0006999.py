def delete_customer(self, customer_id):
		"""Delete a customer."""
		content = self._fetch("/customer/%s" % customer_id, method="DELETE")
		return self._status(content)