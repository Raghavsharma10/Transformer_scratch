def retrieve(payment, refund_id):
        """
        Retrieve a refund from a payment and the refund id.

        :param payment: The payment id or the payment object
        :type payment: resources.Payment|string
        :param refund_id: The refund id
        :type refund_id: string

        :return: The refund resource
        :rtype: resources.Refund
        """
        if isinstance(payment, resources.Payment):
            payment = payment.id

        http_client = HttpClient()
        response, _ = http_client.get(routes.url(routes.REFUND_RESOURCE, resource_id=refund_id, payment_id=payment))
        return resources.Refund(**response)