def remember(self, request, username, **kw):
        """ Returns 'WWW-Authenticate' header with a value that should be used
        in 'Authorization' header.
        """
        if self.credentials_callback:
            token = self.credentials_callback(username, request)
            api_key = 'ApiKey {}:{}'.format(username, token)
            return [('WWW-Authenticate', api_key)]