def update_customer(self, customer_id, **kwargs):
		"""Update a customer."""
		body = self._formdata(kwargs, FastlyCustomer.FIELDS)
		content = self._fetch("/customer/%s" % customer_id, method="PUT", body=body)
		return FastlyCustomer(self, content)