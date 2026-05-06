def _makeService(self):
        """
        Construct a service for the endpoint as described.
        """
        if self._endpointService is None:
            _service = service
        else:
            _service = self._endpointService
        return _service(
            self.description.encode('ascii'), self.factory.getFactory())