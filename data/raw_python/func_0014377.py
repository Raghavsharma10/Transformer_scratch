def create(self, resource_id=None, attributes=None):
        """
        Creates a resource with the given ID (optional) and attributes.
        """

        if attributes is None:
            attributes = {}

        result = None
        if not resource_id:
            result = self.client._post(
                self._url(resource_id),
                self._resource_class.create_attributes(attributes),
                headers=self._resource_class.create_headers(attributes)
            )
        else:
            result = self.client._put(
                self._url(resource_id),
                self._resource_class.create_attributes(attributes),
                headers=self._resource_class.create_headers(attributes)
            )

        return result