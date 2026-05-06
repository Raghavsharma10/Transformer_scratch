def abort(payment):
        """
        Abort a payment from its id.

        :param payment: The payment id or payment object
        :type payment: string|Payment

        :return: The payment resource
        :rtype: resources.Payment
        """
        if isinstance(payment, resources.Payment):
            payment = payment.id

        http_client = HttpClient()
        response, __ = http_client.patch(routes.url(routes.PAYMENT_RESOURCE, resource_id=payment), {'abort': True})
        return resources.Payment(**response)