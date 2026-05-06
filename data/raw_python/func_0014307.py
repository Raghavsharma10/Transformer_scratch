def create(self, attributes=None, **kwargs):
        """
        Creates a webhook with given attributes.
        """

        return super(WebhooksProxy, self).create(resource_id=None, attributes=attributes)