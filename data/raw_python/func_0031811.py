def get(self, resource, operation_timeout=None, max_envelope_size=None, locale=None):
        """
        resource can be a URL or a ResourceLocator
        """
        if isinstance(resource, str):
            resource = ResourceLocator(resource)

        headers = self._build_headers(resource, Session.GetAction, operation_timeout, max_envelope_size, locale)
        self.service.invoke.set_options(tsoapheaders=headers)
        return self.service.invoke