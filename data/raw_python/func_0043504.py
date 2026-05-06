def create(payment, **data):
        """
        Create a refund on a payment.

        :param payment: Either the payment object or the payment id you want to refund.
        :type payment: resources.Payment|string
        :param data: data required to create the refund

        :return: The refund resource
        :rtype resources.Refund
        """
        if isinstance(payment, resources.Payment):
            payment = payment.id

        http_client = HttpClient()
        response, _ = http_client.post(routes.url(routes.REFUND_RESOURCE, payment_id=payment), data)
        return resources.Refund(**response)