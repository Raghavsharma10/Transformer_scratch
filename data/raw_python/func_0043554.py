def get_consistent_resource(self):
        """
        :return a refund that you can trust.
        :rtype Refund
        """
        http_client = HttpClient()
        response, _ = http_client.get(
            routes.url(routes.REFUND_RESOURCE, resource_id=self.id, payment_id=self.payment_id)
        )
        return Refund(**response)