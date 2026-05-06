def update(customer, **data):
        """
        Update a customer from its id.

        :param customer: The customer id or object
        :type customer: string|Customer
        :param data: The data you want to update

        :return: The customer resource
        :rtype resources.Customer
        """
        if isinstance(customer, resources.Customer):
            customer = customer.id

        http_client = HttpClient()
        response, _ = http_client.patch(routes.url(routes.CUSTOMER_RESOURCE, resource_id=customer), data)
        return resources.Customer(**response)