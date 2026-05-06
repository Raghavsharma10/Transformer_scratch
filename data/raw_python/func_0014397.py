def create(self, resource_id=None, attributes=None):
        """
        Creates a resource with a given ID (optional) and attributes for the current content type.
        """

        return self.proxy.create(resource_id=resource_id, attributes=attributes)