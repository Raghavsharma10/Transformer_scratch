def create(**data):
        """
        Create a customer.

        :param data: data required to create the customer

        :return: The customer resource
        :rtype resources.Customer
        """
        http_client = HttpClient()
        response, _ = http_client.post(routes.url(routes.CUSTOMER_RESOURCE), data)
        return resources.Customer(**response)