def delete(customer, card):
        """
        Delete a card from its id.

        :param customer: The customer id or object
        :type customer: string|Customer
        :param card: The card id or object
        :type card: string|Card
        """
        if isinstance(customer, resources.Customer):
            customer = customer.id
        if isinstance(card, resources.Card):
            card = card.id

        http_client = HttpClient()
        http_client.delete(routes.url(routes.CARD_RESOURCE, resource_id=card, customer_id=customer))