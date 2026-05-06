def set_public_limits(self):
        """ Set public limits if auth is enabled and user is not
        authenticated.

        Also sets default limit for GET, HEAD requests.
        """
        if self.request.method.upper() in ['GET', 'HEAD']:
            self._query_params.process_int_param('_limit', 20)
        if self._auth_enabled and not getattr(self.request, 'user', None):
            wrappers.set_public_limits(self)