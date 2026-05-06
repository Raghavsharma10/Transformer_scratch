def list_cards(self, *args, **kwargs):
        """
        List the cards of the customer.

        :param page: the page number
        :type page: int|None
        :param per_page: number of customers per page. It's a good practice to increase this number if you know that you
        will need a lot of payments.
        :type per_page: int|None
        :return: The cards of the customer
        :rtype APIResourceCollection
        """
        return payplug.Card.list(self, *args, **kwargs)