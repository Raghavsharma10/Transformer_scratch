def retrieve(customer_id):
        """
        Retrieve a customer from its id.

        :param customer_id: The customer id
        :type customer_id: string

        :return: The customer resource
        :rtype: resources.Customer
        """
        http_client = HttpClient()
        response, __ = http_client.get(routes.url(routes.CUSTOMER_RESOURCE, resource_id=customer_id))
        return resources.Customer(**response)