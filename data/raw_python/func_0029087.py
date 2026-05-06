def _discover(self):
        """Discovers methods in the XML-RPC API and creates attributes for them
        on this object. Enables stuff like "magento.cart.create(...)" to work
        without having to define Python methods for each XML-RPC equivalent.
        """

        self._resources = {}
        resources = self._client.resources(self._session_id)
        for resource in resources:
            self._resources[resource['name']] = MagentoResource(
                self._client, self._session_id, resource['name'],
                resource['title'], resource['methods'])