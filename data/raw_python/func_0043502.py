def list(per_page=None, page=None):
        """
        List of payments. You have to handle pagination manually

        :param page: the page number
        :type page: int|None
        :param per_page: number of payment per page. It's a good practice to increase this number if you know that you
        will need a lot of payments.
        :type per_page: int|None

        :return A collection of payment
        :rtype resources.APIResourceCollection
        """
        # Comprehension dict are not supported in Python 2.6-. You can use this commented line instead of the current
        # line when you drop support for Python 2.6.
        # pagination = {key: value for (key, value) in [('page', page), ('per_page', per_page)] if value}
        pagination = dict((key, value) for (key, value) in [('page', page), ('per_page', per_page)] if value)

        http_client = HttpClient()
        response, _ = http_client.get(routes.url(routes.PAYMENT_RESOURCE, pagination=pagination))
        return resources.APIResourceCollection(resources.Payment, **response)