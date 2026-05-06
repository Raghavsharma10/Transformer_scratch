def create(customer, **data):
        """
        Create a card instance.

        :param customer: the customer id or object
        :type customer: string|Customer
        :param data: data required to create the card

        :return: The card resource
        :rtype resources.Card
        """
        if isinstance(customer, resources.Customer):
            customer = customer.id

        http_client = HttpClient()
        response, _ = http_client.post(routes.url(routes.CARD_RESOURCE, customer_id=customer), data)
        return resources.Card(**response)