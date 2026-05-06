def retrieve(customer, card_id):
        """
        Retrieve a card from its id.

        :param customer: The customer id or object
        :type customer: string|Customer
        :param card_id: The card id
        :type card_id: string

        :return: The customer resource
        :rtype: resources.Card
        """
        if isinstance(customer, resources.Customer):
            customer = customer.id

        http_client = HttpClient()
        response, __ = http_client.get(routes.url(routes.CARD_RESOURCE, resource_id=card_id, customer_id=customer))
        return resources.Card(**response)