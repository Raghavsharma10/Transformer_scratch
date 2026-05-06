def get_consistent_resource(self):
        """
        :return a payment that you can trust.
        :rtype Payment
        """
        http_client = HttpClient()
        response, _ = http_client.get(routes.url(routes.PAYMENT_RESOURCE, resource_id=self.id))
        return Payment(**response)