def put(self, resource, obj,
            operation_timeout=None, max_envelope_size=None, locale=None):
        """
        resource can be a URL or a ResourceLocator
        """
        headers = None
        return self.service.invoke(headers, obj)