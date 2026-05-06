def list(payment):
        """
        List all the refunds for a payment.

        :param payment: The payment object or the payment id
        :type payment: resources.Payment|string

        :return: A collection of refunds
        :rtype resources.APIResourceCollection
        """
        if isinstance(payment, resources.Payment):
            payment = payment.id

        http_client = HttpClient()
        response, _ = http_client.get(routes.url(routes.REFUND_RESOURCE, payment_id=payment))
        return resources.APIResourceCollection(resources.Refund, **response)