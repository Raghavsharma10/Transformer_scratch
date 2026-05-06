def create(**data):
        """
        Create a Payment request.

        :param data: data required to create the payment

        :return: The payment resource
        :rtype resources.Payment
        """
        http_client = HttpClient()
        response, _ = http_client.post(routes.url(routes.PAYMENT_RESOURCE), data)
        return resources.Payment(**response)