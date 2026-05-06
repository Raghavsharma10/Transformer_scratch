def retrieve(payment_id):
        """
        Retrieve a payment from its id.

        :param payment_id: The payment id
        :type payment_id: string

        :return: The payment resource
        :rtype: resources.Payment
        """
        http_client = HttpClient()
        response, __ = http_client.get(routes.url(routes.PAYMENT_RESOURCE, resource_id=payment_id))
        return resources.Payment(**response)