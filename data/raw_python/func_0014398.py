def find(self, resource_id, query=None):
        """
        Finds a single resource by ID related to the current space.
        """

        return self.proxy.find(resource_id, query=query)