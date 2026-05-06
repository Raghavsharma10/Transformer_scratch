def recieve(self, resource, obj,
                operation_timeout=None, max_envelope_size=None, locale=None):
        """
        resource can be a URL or a ResourceLocator
        """
        if isinstance(resource, str):
            resource = ResourceLocator(resource)

        headers = self._build_headers(resource, Session.ReceiveAction,
                                      operation_timeout, max_envelope_size, locale)
        return self.service.invoke(headers, obj)