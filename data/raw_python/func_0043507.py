def delete(customer):
        """
        Delete a customer from its id.

        :param customer: The customer id or object
        :type customer: string|Customer
        """
        if isinstance(customer, resources.Customer):
            customer = customer.id

        http_client = HttpClient()
        http_client.delete(routes.url(routes.CUSTOMER_RESOURCE, resource_id=customer))